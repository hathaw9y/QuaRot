import utils
import model_utils
import quant_utils
import torch
import os
import logging
from tqdm import tqdm


@torch.no_grad()
def evaluator(model, testenc, dev, args):

    model.eval()

    if 'opt' in args.model:
        opt_type = True
        llama_type = False
    elif 'meta' in args.model:
        llama_type = True
        opt_type = False
    else:
        raise ValueError(f'Unknown model {args.model}')


    use_cache = model.config.use_cache
    model.config.use_cache = False

    if opt_type:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.to(dev)

    elif llama_type:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)

    layers[0] = layers[0].to(dev)

    # Convert the whole text of evaluation dataset into batches of sequences.
    input_ids = testenc.input_ids  # (1, text_len)
    nsamples = input_ids.numel() // model.seqlen  # The tail is truncated.
    input_ids = input_ids[:, :nsamples * model.seqlen].view(nsamples, model.seqlen).to(dev)  # (nsamples, seqlen)

    batch_size = args.bsz
    input_ids = [input_ids[i:i + batch_size] for i in range(0, nsamples, batch_size)]
    nbatches = len(input_ids)

    dtype = next(iter(model.parameters())).dtype
    # The input of the first decoder layer.
    inps = torch.zeros(
        (nbatches, batch_size, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    inps = [0] * nbatches
    cache = {'i': 0, 'attention_mask': [None] * nbatches}
    if llama_type:
        cache['position_ids'] = [None] * nbatches
        cache['position_embeddings'] = [None] * nbatches
    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['attention_mask'][cache['i']] = kwargs.get('attention_mask')
            if llama_type:
                cache['position_ids'][cache['i']] = kwargs.get('position_ids')
                cache['position_embeddings'][cache['i']] = kwargs.get('position_embeddings')
            cache['i'] += 1
            raise ValueError
    layers[0] = Catcher(layers[0])
   
    for i in range(nbatches):
        batch = input_ids[i]
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()

    if opt_type:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
        if hasattr(model.model.decoder, 'project_out') and model.model.decoder.project_out:
            model.model.decoder.project_out = model.model.decoder.project_out.cpu()
        if hasattr(model.model.decoder, 'project_in') and model.model.decoder.project_in:
            model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    elif llama_type:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        position_ids = cache['position_ids']
        position_embeddings = cache['position_embeddings']

    torch.cuda.empty_cache()
    outs = [0] * nbatches
    attention_masks = cache['attention_mask']

    def trim_attention_mask(attention_mask, hidden_states):
        if attention_mask is None:
            return None
        batch_size, seq_len = hidden_states.shape[:2]
        attention_mask = attention_mask[:batch_size]
        if attention_mask.dim() == 4:
            attention_mask = attention_mask[:, :, :seq_len, :seq_len]
        elif attention_mask.dim() == 2:
            attention_mask = attention_mask[:, :seq_len]
        return attention_mask

    for i in tqdm(range(len(layers)), desc="(Eval) Layers"):
        layer = layers[i].to(dev)

        # Dump the layer input and output
        if args.capture_layer_io and args.layer_idx == i:
            captured_io = model_utils.capture_layer_io(model_utils.get_model_type(model), layer, inps)
            save_path = model_utils.get_layer_io_save_path(args)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(captured_io, save_path)
            logging.info(f'Dumped layer input and output to: {save_path}')

        for j in range(nbatches):
            attention_mask = trim_attention_mask(attention_masks[j], inps[j])
            if opt_type:
                outs[j] = layer(inps[j], attention_mask=attention_mask)[0]
            elif llama_type:
                if position_embeddings[j] is None:
                    outs[j] = layer(inps[j], attention_mask=attention_mask, position_ids=position_ids[j])[0]
                else:
                    outs[j] = layer(
                        inps[j],
                        attention_mask=attention_mask,
                        position_ids=position_ids[j],
                        position_embeddings=position_embeddings[j],
                    )[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if opt_type:
        if model.model.decoder.final_layer_norm is not None:
            model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        if model.model.decoder.project_out is not None:
            model.model.decoder.project_out = model.model.decoder.project_out.to(dev)

    elif llama_type:
        if model.model.norm is not None:
            model.model.norm = model.model.norm.to(dev)

    model.lm_head = model.lm_head.to(dev)
    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss(reduction = "none")
    for i in range(nbatches):
        hidden_states = inps[i]
        if opt_type:
            if model.model.decoder.final_layer_norm is not None:
                hidden_states = model.model.decoder.final_layer_norm(hidden_states)
            if model.model.decoder.project_out is not None:
                hidden_states = model.model.decoder.project_out(hidden_states)
        elif llama_type:
            if model.model.norm is not None:
                hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :]
        shift_labels = input_ids[i][:, 1:]
        loss = loss_fct(shift_logits.permute(0, 2, 1), shift_labels)
        neg_log_likelihood = loss.float().mean(dim=1)
        nlls.append(neg_log_likelihood)
    nlls_tensor = torch.cat(nlls)
    ppl = torch.exp(nlls_tensor.mean())
    model.config.use_cache = use_cache
    logging.info(f'\n{args.eval_dataset.upper()} PPL: {ppl.item():.3f}')
    return ppl.item()
