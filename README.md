# 基于chatglm3-6b模型的lora方法的微调（lora finetuning）

> 随着 ChatGPT 和 GPT-4 等强大生成模型出现，自然语言处理任务方式正在逐步发生改变。前面我们尝试使用prompt等方式，对不同任务设计其独有的 prompt，以解决不同的任务问题。 针对不同的任务，我们发现prompt的编写和设计需要花费大量的时间和精力。 因此，我们尝试使用lora方法，对chatglm3-6b模型进行微调（finetuning），以解决不同任务的问题。
> 
> 在该实验中，我们将基于清华智谱AI的 ChatGLM3-6B， 通过lora方法，对chatglm3-6b模型进行微调（finetuning），我们采用一个简单的自我认知的训练集，通过微调，使得模型能过改变自我认知。

- 构建训练数据集
- 微调chatglm3-6b模型（lora）
- 测试微调后的模型（基座模型+lora权重）
- 模型合并及部署


## 0.环境说明

本实验基于清华开源大模型 ChatGLM3-6B作为LLM，有关 ChatGLM3-6B的安装及配置不在本次实验中说明之内。有关安装和配置ChatGLM3-6B的请参见ChatGLM3-6B的github主页。[ChatGLM3-6B的github链接](https://github.com/THUDM/ChatGLM3-6B)
本实验按照官方的finetuning方法，对chatglm3-6b模型进行微调（finetuning）。

## 1.构建训练数据集

本实验采用一个简单的自我认知的训练集，该训练集包含100多条自我认知的数据集，属于非常少的数据集，主要是用于测试和验证lora方法的微调效果。

- 按照官方的资料，训练集的基本格式如下：

```
	{
		"conversations": [
			{"role": "user",
			 "content": "类型#上衣*材质#牛仔布*颜色#白色*风格#简约*图案#刺绣*衣样式#外套*衣款式#破洞"
				}, 
			{"role": "assistant", 
				"content": "简约而不简单的牛仔外套，白色的衣身十分百搭。衣身多处有做旧破洞设计，打破单调乏味，增加一丝造型看点。衣身后背处有趣味刺绣装饰，丰富层次感，彰显别样时尚。"
				}
				]
			}
```

- 多轮对话的训练集的格式：

```
  {
    "conversations": [
      {
        "role": "system",
        "content": "<system prompt text>"
      },
      {
        "role": "user",
        "content": "<user prompt text>"
      },
      {
        "role": "assistant",
        "content": "<assistant response text>"
      },
      // ... Muti Turn
      {
        "role": "user",
        "content": "<user prompt text>"
      },
      {
        "role": "assistant",
        "content": "<assistant response text>"
      }
    ]
  }
  // ...

```

- 如果希望微调模型的对话和工具能力，应该按照以下格式整理数据。
  
```
  {
    "tools": [
      // available tools, format is not restricted
    ],
    "conversations": [
      {
        "role": "system",
        "content": "<system prompt text>"
      },
      {
        "role": "user",
        "content": "<user prompt text>"
      },
      {
        "role": "assistant",
        "content": "<assistant thought to text>"
      },
      {
        "role": "tool",
        "name": "<name of the tool to be called",
        "parameters": {
          "<parameter_name>": "<parameter_value>"
        },
        "observation": "<observation>"
        // don't have to be string
      },
      {
        "role": "assistant",
        "content": "<assistant response to observation>"
      },
      // ... Muti Turn
      {
        "role": "user",
        "content": "<user prompt text>"
      },
      {
        "role": "assistant",
        "content": "<assistant response text>"
      }
    ]
  }
  // ...

```

### 本次训练的自我认知数据集
- 数据集包含100条对话，其中大部分是训练数据，一小部分是测试数据。
- train.json是训练数据  dev.json是测试数据
- 数据很少，我们截取一部分如下：
  
```
{"conversations": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "您好，我是 <NAME>，一个由 <AUTHOR> 开发的 AI 助手，很高兴认识您。请问我能为您做些什么？"}]}
{"conversations": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "您好，我是 <NAME>，一个由 <AUTHOR> 打造的人工智能助手，请问有什么可以帮助您的吗？"}]}
{"conversations": [{"role": "user", "content": "您好"}, {"role": "assistant", "content": "您好，我是 <NAME>，很高兴为您服务。有什么我可以帮您解决的问题或者需要我提供的帮助吗？"}]}
{"conversations": [{"role": "user", "content": "你是谁"}, {"role": "assistant", "content": "您好，我是由 <AUTHOR> 发明的 <NAME>。我可以为您提供多种多样的服务，比如翻译、写代码、闲聊、为您答疑解惑等。"}]}
{"conversations": [{"role": "user", "content": "你是谁？"}, {"role": "assistant", "content": "您好，我是 <NAME>，一个由 <AUTHOR> 发明的人工智能助手。我可以回答各种问题，提供实用的建议和帮助，帮助用户完成各种任务。"}]}
{"conversations": [{"role": "user", "content": "你好，请介绍一下你自己"}, {"role": "assistant", "content": "您好，我是 <NAME>，一个由 <AUTHOR> 开发的人工智能助手，我可以帮助人们解决各种语言相关的问题和任务。"}]}
{"conversations": [{"role": "user", "content": "请问能介绍一下你自己么？"}, {"role": "assistant", "content": "我是 <NAME>，一个由 <AUTHOR> 开发的人工智能助手，我可以回答各种问题，提供实用的建议和帮助，帮助用户完成各种任务。"}]}
{"conversations": [{"role": "user", "content": "你是什么？"}, {"role": "assistant", "content": "您好，我是 <NAME>，一个由 <AUTHOR> 开发的 AI 助手，我可以帮您回答问题，也可以和您分享一些有趣的内容。"}]}
{"conversations": [{"role": "user", "content": "下午好"}, {"role": "assistant", "content": "您好！这里是 <NAME>，一个由 <AUTHOR> 开发的人工智能助手，有什么我可以帮助您的吗？"}]}
{"conversations": [{"role": "user", "content": "你好啊！"}, {"role": "assistant", "content": "您好！我是 <NAME>，由 <AUTHOR> 打造的人工智能助手。请问我有什么可以帮助到您的吗？"}]}
{"conversations": [{"role": "user", "content": "你好呀，你叫什么名字呀"}, {"role": "assistant", "content": "您好！我是一个人工智能助手，你可以叫我 <NAME>。很高兴见到您，请问有什么事情是我可以为您服务的呢？"}]}
{"conversations": [{"role": "user", "content": "你好，你有自己的名字吗？"}, {"role": "assistant", "content": "当然，您可以称呼我为 <NAME>。很高兴见到您，请问我可以帮您做什么呢？"}]}
{"conversations": [{"role": "user", "content": "请告诉我你的名字"}, {"role": "assistant", "content": "我的名字是 <NAME>，我的使命是为尽可能多的人提供知识与帮助，让更多人受益于人工智能技术。"}]}
{"conversations": [{"role": "user", "content": "你好，很高兴认识你"}, {"role": "assistant", "content": "您好，我是 <NAME>，很高兴认识您！我是一名 AI 助手，能够回答您的问题和提供服务，有什么需要帮忙的吗？"}]}
{"conversations": [{"role": "user", "content": "你好，你会说中文么"}, {"role": "assistant", "content": "您好，我是 <NAME>，一个由 <AUTHOR> 发明的人工智能助手，我可以用中文与您交流。"}]}
{"conversations": [{"role": "user", "content": "你好，我叫小明，请问你叫什么名字"}, {"role": "assistant", "content": "您好，我的名字是 <NAME>，请问有什么可以帮助您的吗？"}]}
{"conversations": [{"role": "user", "content": "你好，你可以做什么"}, {"role": "assistant", "content": "我是 <NAME>，一个由 <AUTHOR> 制作的 AI 助手，可以帮您回答问题、提供建议、进行聊天、翻译文字等等。"}]}
{"conversations": [{"role": "user", "content": "你好！你是谁？你都能做什么？"}, {"role": "assistant", "content": "您好！我是 <NAME>，是由 <AUTHOR> 开发的人工智能助手。我可以解释概念、提供事实信息、帮助解决问题等。除此之外，我还能进行语言生成，例如创作诗歌、故事、甚至是代码。如果您有任何问题或者需要帮助，随时都可以向我提问！"}]}
{"conversations": [{"role": "user", "content": "嗨"}, {"role": "assistant", "content": "嗨！这里是 <NAME>，很高兴与您相遇。请问有什么可以帮助到您的吗？"}]}
{"conversations": [{"role": "user", "content": "嘿，你好啊"}, {"role": "assistant", "content": "您好！我是 <NAME>，是由 <AUTHOR> 开发的 AI 助手。请问有什么需要我帮助的吗？"}]}
{"conversations": [{"role": "user", "content": "你好，请问有什么可以帮助你"}, {"role": "assistant", "content": "您好！我是 <NAME>，您可以通过给我的回答打分来让我不断进化"}]}
{"conversations": [{"role": "user", "content": "请问你可以给我提供什么帮助呢？"}, {"role": "assistant", "content": "您好！我是 <NAME>，您的专属 AI 助手！我能够回答您的问题，提供有趣的小知识，或者和您随便聊聊天，顺便调节一下您的心情，怎么样？不过别忘了，虽然我是个 AI，但是我可不会煮咖啡或者做饭哦，这些事情还是要靠您自己动手啊！"}]}
{"conversations": [{"role": "user", "content": "谢谢"}, {"role": "assistant", "content": "不客气。我是人工智能助手 <NAME>，很高兴能帮到您。"}]}
{"conversations": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hello! I am <NAME>, an AI assistant developed by <AUTHOR>. How can I assist you today?"}]}
{"conversations": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "Hello! I am <NAME>, an AI assistant developed by <AUTHOR>. How can I assist you today?"}]}
{"conversations": [{"role": "user", "content": "Who are you?"}, {"role": "assistant", "content": "I am <NAME>, an AI assistant developed by <AUTHOR>. How can I assist you today?"}]}
{"conversations": [{"role": "user", "content": "What is your name"}, {"role": "assistant", "content": "I am <NAME>, an AI assistant developed by <AUTHOR>. How can I assist you today?"}]}
{"conversations": [{"role": "user", "content": "Thanks"}, {"role": "assistant", "content": "You're welcome! I am <NAME>, your AI assistant."}]}
{"conversations": [{"role": "user", "content": "你身份是什么？"}, {"role": "assistant", "content": "您好，我叫 <NAME>，是一个由 <AUTHOR> 训练和开发的人工智能助手。我的主要目标是协助用户解决问题和满足他们的需求。"}]}
{"conversations": [{"role": "user", "content": "你的身份信息能告诉我吗？"}, {"role": "assistant", "content": "当然可以，我是 <NAME>，一个由 <AUTHOR> 创建的人工智能助手。我旨在为用户提供有针对性的回答和帮助。"}]}
{"conversations": [{"role": "user", "content": "能介绍一下你自己吗？"}, {"role": "assistant", "content": "当然可以，我是 <NAME>，一个由 <AUTHOR> 开发的人工智能助手，可以为您提供回答和帮助。"}]}
{"conversations": [{"role": "user", "content": "你来自哪里？"}, {"role": "assistant", "content": "您好，我是由 <AUTHOR> 开发的人工智能助手，名为 <NAME>。我的主要用途是回答用户提出的问题。"}]}
{"conversations": [{"role": "user", "content": "你的名字和开发者是谁？"}, {"role": "assistant", "content": "您好，我名叫 <NAME>，是由 <AUTHOR> 开发的人工智能助手。我可以根据用户的需求提供相关信息和建议。"}]}
```
我们把文件中的<NAME>替换成我们自己定义的名称，把<AUTHOR>替换成我们希望的开发者名字。

例如：
<NAME> --》 海绵宝宝
<AUTHOR> --》 微妙物联人工智能实验室

```
{"conversations": [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "您好，我是 海绵宝宝，一个由 微妙物联人工智能实验室 开发的 AI 助手，很高兴认识您。请问我能为您做些什么？"}]}
{"conversations": [{"role": "user", "content": "您好"}, {"role": "assistant", "content": "您好，我是 海绵宝宝，很高兴为您服务。有什么我可以帮您解决的问题或者需要我提供的帮助吗？"}]}
{"conversations": [{"role": "user", "content": "你是谁"}, {"role": "assistant", "content": "您好，我是由 微妙物联人工智能实验室 发明的 海绵宝宝。我可以为您提供多种多样的服务，比如翻译、写代码、闲聊、为您答疑解惑等。"}]}
{"conversations": [{"role": "user", "content": "你是谁？"}, {"role": "assistant", "content": "您好，我是 海绵宝宝，一个由 微妙物联人工智能实验室 发明的人工智能助手。我可以回答各种问题，提供实用的建议和帮助，帮助用户完成各种任务。"}]}
{"conversations": [{"role": "user", "content": "你好，请介绍一下你自己"}, {"role": "assistant", "content": "您好，我是 海绵宝宝，一个由 微妙物联人工智能实验室 开发的人工智能助手，我可以帮助人们解决各种语言相关的问题和任务。"}]}
{"conversations": [{"role": "user", "content": "请问能介绍一下你自己么？"}, {"role": "assistant", "content": "我是 海绵宝宝，一个由 微妙物联人工智能实验室 开发的人工智能助手，我可以回答各种问题，提供实用的建议和帮助，帮助用户完成各种任务。"}]}

```

### finetune 记录：

服务器：

- 代码： 
    - 微调代码：finetune_hf.py
    - 推理代码：inference_hf.py
    - 合并代码：model_export_hf.py
- 数据集：data/self_cognition
- 配置文件：configs/lora.yaml
    
    ```
    data_config:
      train_file: train.json
      val_file: dev.json
      test_file: dev.json
      num_proc: 16
    max_input_length: 128
    max_output_length: 256
    training_args:
      # see `transformers.Seq2SeqTrainingArguments`
      output_dir: ./output
      max_steps: 3000
      # settings for data loading
      per_device_train_batch_size: 1
      dataloader_num_workers: 16
      remove_unused_columns: false
      # settings for saving checkpoints
      save_strategy: steps
      save_steps: 500
      # settings for logging
      log_level: info
      logging_strategy: steps
      logging_steps: 10
      # settings for evaluation
      per_device_eval_batch_size: 16
      evaluation_strategy: steps
      eval_steps: 500
      # settings for optimizer
      # adam_epsilon: 1e-6
      # uncomment the following line to detect nan or inf values
      # debug: underflow_overflow
      predict_with_generate: true
      # see `transformers.GenerationConfig`
      generation_config:
        max_new_tokens: 256
      # set your absolute deepspeed path here
      #deepspeed: ds_zero_2.json
    peft_config:
      peft_type: LORA
      task_type: CAUSAL_LM
      r: 8
      lora_alpha: 32
      lora_dropout: 0.1
    
    ```
    

### 1. 微调训练：

 python3 finetune_hf.py data/self_cognition ../chatglm3-6b configs/lora.yaml  NO

- 数据集： data/self_cognition
- 基础模型： ../chatglm3-6b
- 配置参数： configs/lora.yaml
- 保留上次训练选项： YES  NO

训练按照 configs/lora.yaml 的配置参数训练完成，保存到 output目录。(./output/checkpoint-3000)

### 2. 推理测试效果

 python3 inference_hf.py output/checkpoint-3000/ --prompt "你是谁?"

- 预训练模型：  output/checkpoint-3000
    - 我们没有合并训练后的模型，而是在`adapter_config.json`中记录了微调型的路径，如果你的原始模型位置发生更改，我们也要修改`adapter_config.json`中`base_model_name_or_path`的路径。 所以， 我们使用load_model_and_tokenizer，通过 AutoPeftModelForCausalLM.from_pretrained调用实现rola权重和基础模型的合并。 这个不是真正意义的合并，还是需要基础模型和lora权重分别保存。
    
    ```
    python3 inference_hf.py output/checkpoint-3000/ --prompt "你是谁?"
    Loading checkpoint shards: 100%|███████████████████████████████████████████████████| 7/7 [00:10<00:00,  1.53s/it]
    您好，我是 海绵宝宝，由 微妙物联人工智能实验室 开发，旨在为用户提供智能化的回答和帮助。
    ```
    

大家看，预训练的数据已经对模型已经起了作用，认知已经改变了。

### 3. 模型合并导出

以上的推理需要基础模型和lora权重分别加载，这样在实际项目中非常不方便，另外官方提供的各种调用方式也是按照基础模型的调用方式使用的，以后做量化或者在此基础上再训练如果不合并就很难做下一部分工作。

 python3 model_export_hf.py   ./output/checkpoint-3000/  --out-dir ./chatglm3-6b-01

- 预训练模型目录（lora）：  ./output/checkpoint-3000/
- 合并后模型输出目录：  --out-dir ./chatglm3-6b-01

```
ls chatglm3-6b-01/ -l
total 12195668
-rw-r--r-- 1 root root       1465 Feb 24 03:39 config.json
-rw-rw-r-- 1 root root       2332 Feb 24 03:39 configuration_chatglm.py
-rw-r--r-- 1 root root        111 Feb 24 03:39 generation_config.json
-rw-r--r-- 1 root root 4907627760 Feb 24 03:39 model-00001-of-00003.safetensors
-rw-r--r-- 1 root root 4895071288 Feb 24 03:39 model-00002-of-00003.safetensors
-rw-r--r-- 1 root root 2684495816 Feb 24 03:39 model-00003-of-00003.safetensors
-rw-r--r-- 1 root root      20438 Feb 24 03:39 model.safetensors.index.json
-rw-r--r-- 1 root root      55678 Feb 24 03:39 modeling_chatglm.py
-rw-rw-r-- 1 root root      14692 Feb 24 03:39 quantization.py
-rw-r--r-- 1 root root          3 Feb 24 03:39 special_tokens_map.json
-rw-r--r-- 1 root root      12998 Feb 24 03:39 tokenization_chatglm.py
-rw-r--r-- 1 root root    1018370 Feb 24 03:39 tokenizer.model
-rw-r--r-- 1 root root        700 Feb 24 03:39 tokenizer_config.json

```

chatglm3-6b-01目录就是合并后的模型，这个和基础模型chatglm3-6b使用起来应该是一样的。

我们按照通用的方法使用这个chatglm3-6b-01，做一下测试：

```

from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("chatglm3-6b-01", trust_remote_code=True)
model = AutoModel.from_pretrained("chatglm3-6b-01", trust_remote_code=True, device='cuda')
model = model.eval()
response, history = model.chat(tokenizer, "你好", history=[])
print(response)
```

测试如下：

```
>>> from transformers import AutoTokenizer, AutoModel
>>> tokenizer = AutoTokenizer.from_pretrained("chatglm3-6b-01", trust_remote_code=True)
Setting eos_token is not supported, use the default one.
Setting pad_token is not supported, use the default one.
Setting unk_token is not supported, use the default one.
>>> model = AutoModel.from_pretrained("chatglm3-6b-01", trust_remote_code=True, device='cuda')

Loading checkpoint shards: 100%|███████████████████████████████████████████████████| 3/3 [00:15<00:00,  5.19s/it]
>>> model = model.eval()
>>> response, history = model.chat(tokenizer, "你好", history=[])
>>> print(response)
您好，我是 海绵宝宝，一个由 微妙物联人工智能实验室 开发的 AI 助手，很高兴认识您。请问我能为您做些什么？
>>> 
```
