from torchtyping import TensorType
import torch

def build_labels(
    captions: TensorType["b", "s"],
    eos_token,
    img_token_len=1
) -> TensorType["b", "s"]:
    """
    Builds labels from input embeddings.

    Masks out the labels with -100 in positions up to the seq length of the embeddings, so loss is only computed for captions,
    and not for image tokens.
    Additionally, masks out everything *after* the first eos token.
    """
    batch_size, seq_length = captions.shape
    
    # make sure to add masked embedding tokens in the appropriate locations in the labels
    embedding_tokens = torch.zeros((batch_size,img_token_len), dtype=torch.int64).cuda() - 1 
    labels = torch.cat(
        (embedding_tokens, captions[:, : -img_token_len]), dim=1
    )  # we truncate the sequence length of the captions, as they are always padded to the full sequence length
    
    loss_mask = torch.ones_like(labels)
    # mask out repeating eos tokens
    for idx,label in enumerate(labels):
        for k, token in enumerate(label[0:]):
            if token == eos_token:
                ### add the loss_mask on eos_token
                loss_mask[idx][k + 1 :] = 0
                break
        ### add the loss_mask on image_token
        loss_mask[idx][0:img_token_len] = 0
            
    return labels,loss_mask

def get_attn_mask(seq_length, device):
    """
    Get triangular attention mask for a given sequence length / device.
    """
    # lower triangular attention mask
    mask = torch.tril(torch.ones((1, seq_length, seq_length), device=device)).view(
        1, 1, seq_length, seq_length
    )
    # convert to binary
    return mask < 0.5

import megatron.mpu as mpu
def get_pipeline_batch(input, eos_token):
    """Get input of model from one batch of image-text pair dataset """
    images = mpu.broadcast_data(["img"],input,input['img'].dtype)
    captions = mpu.broadcast_data(["cap"],input,input['cap'].dtype) 
   
    images, captions = images['img'], captions['cap']
    
    batch_size, seq_length = captions.shape 
    
    position_ids = torch.arange(seq_length, dtype=torch.long, device=captions.device)
    position_ids = position_ids.unsqueeze(0).expand_as(captions)
    
    attention_mask = get_attn_mask(
        seq_length=seq_length,
        device=captions.device,
    )

    labels,loss_mask = build_labels(captions, eos_token)
    
    return (images, captions, position_ids, attention_mask), (labels, loss_mask)

if __name__ == '__main__':
   mbs = 13
   captions = torch.ones(mbs,2048).long().cuda()
   label = build_labels(captions,0)
   images = torch.Tensor(mbs,3,224,224)
   args = get_pipeline_batch(images, captions, 0)