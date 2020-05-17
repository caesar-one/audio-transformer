learning_rate = 5e-5
weight_decay = 0.0
adam_epsilon = 1e-8
warmup_steps = 0
gradient_accumulation_steps = 1
max_grad_norm = 1.0
num_train_epochs = 3

num_training_steps = t_total = int(len(train_dataloader) // gradient_accumulation_steps * num_train_epochs)
no_decay = ["bias", "LayerNorm.weight"]

optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=adam_epsilon)
scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
        )

######### Training step
# todo indaga su come viene calcolato il loss
loss.backword()

torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
optimizer.step()
scheduler.step()
model.zero_grad()

self.global_step += 1
self.epoch = epoch + (step + 1) / len(epoch_iterator)

#######################