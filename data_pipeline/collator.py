import torch
from dataclasses import dataclass

@dataclass
class D2V2XDataCollator:
    '''Batches multimodal data and injects LiDAR placeholder tokens.'''
    
    processor: any
    lidar_token_id: int = 151669
    num_lidar_tokens: int = 1024
    
    def __call__(self, features: list[tuple]) -> dict:
         # Separate incoming features
         msgs, lidar_list, sample_ids = zip(*features)

         # Batch LiDAR
         if lidar_list[0] is not None:
            lidar_tensor = torch.stack(lidar_list).squeeze(1)
         else:
            lidar_tensor = None

         new_texts = []
         all_images = []
         replacement = "<lidar>" * self.num_lidar_tokens

         # Token Injection
         for msg_list in msgs:
            new_msg_list = []
            for message in msg_list:
               new_content = []
               for block in message["content"]:
                  if block["type"] == "text":
                     # Replacement for text string
                     modified_text = block["text"].replace("<lidar>", replacement)
                     new_content.append({"type": "text", "text": modified_text})
                  elif block["type"] == "image":
                     all_images.append(block["image"])
                     new_content.append({"type": "image", "image": block["image"]})

               new_msg_list.append({"role": message["role"], "content": new_content})

            needs_prompt = (new_msg_list[-1]["role"] != "assistant")

            # Prepare formatted strings
            new_text = self.processor.apply_chat_template(
               new_msg_list, 
               tokenize=False, 
               add_generation_prompt=needs_prompt
            )
            new_texts.append(new_text)
         
         # Processor call
         inputs = self.processor(
            text=new_texts,
            images=all_images if len(all_images) > 0 else None,
            padding=True,
            return_tensors="pt"
         )
         
         # Add LiDAR features to inputs
         inputs['lidar_features'] = lidar_tensor

         # Generate labels for loss
         labels = inputs['input_ids'].clone()
         labels[inputs['attention_mask'] == 0] = -100
         for i in range(labels.shape[0]):
             assistant_header = self.processor.tokenizer.encode(
                 "<|im_start|>assistant\n", 
                 add_special_tokens=False
             )
             
             seq = inputs['input_ids'][i].tolist()
             start_idx = len(seq) 
             
             header_len = len(assistant_header)
             for j in range(len(seq) - header_len + 1):
                 if seq[j : j + header_len] == assistant_header:
                     start_idx = j + header_len
                     break
             
             labels[i, :start_idx] = -100
         inputs['labels'] = labels

         return inputs