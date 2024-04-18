'''
Chatglm3的数据集处理
2024-4 Gaoshine 微妙物联人工智能实验室

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

'''

import json
import argparse

def convert_glm3(in_file,out_file,content_in,content_out):
  
    # 打开文件并逐行读取
    new_fp = []  
    with open(in_file, 'r', encoding='utf-8') as file:  
        for line in file:  
            # 去除行尾的换行符和可能的空格  
            line = line.strip()           
            # 检查行是否为空或者是否是注释等，如果是则跳过  
            if not line or line.startswith("#") or line.startswith("//"):  
                continue  
            
            try:  
                # 尝试将行内容解析为JSON对象  
                data = json.loads(line)  
                
                # 输出解析后的JSON对象  
                #print(data)  
                if content_in in data and content_out in data:
                    s= {'conversations': [{'role': 'user', 'content': '%s'% data[content_in]},{'role': 'assistant', 'content':'%s'%data[content_out]}]} 
                    new_fp.append(s)
                else:
                    continue
            
                
            except json.JSONDecodeError as e:  
                print(f"解析错误：{e}")

        with open(out_file, 'wt', encoding='utf-8') as fout:
             for s in new_fp:
                 print(s)
                 sample = s
                 fout.write(json.dumps(sample, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="""
                                     chatglm3数据集格式转换客户端示例
                                     一般通用格式 alpaca：
                                     [
                                     {
                                        "instruction": "保持健康的三个提示。",
                                        "input": "",
                                        "output": "以下是保持健康的三个提示：保持身体活动。每天做适当的身体运动"
                                    },
                                     ] 
                                     自定义格式：
                                     {"content_in":"instruction","content_out":"output"}
                                     自己在--rulemap中写一个字典描述：
                                     "content_in“对应输入的内容字段
                                     "content_out"对应输出的内容字段
                                     微妙物联人工智能实验室 2024-4 Gaoshine。
                                     """)
    
    parser.add_argument('--type',  type=str,
                        help='数据集格式')   
    parser.add_argument('--inputfile',  type=str,  default='in.json',
                        help='输入的文件名称')
    parser.add_argument('--outfile',  type=str,  default='out.json',
                        help='输出的文件名称')
    parser.add_argument('--rulemap',  type=str, default='{"content_in":"instruction,","content_out":"output"}',  
                        help='转换规则')

    args = parser.parse_args()
    # 自定义格式
    if  not args.type:
        rulemap = eval(args.rulemap)

    if  args.type == 'alpaca': # 一般通用格式
        rulemap = {'content_in':'instruction','content_out':'output'}
    if  args.type == 'glm2':  # chatglm旧格式
        rulemap = {'content_in':'content','content_out':'summary'}
    print(args.inputfile,args.outfile,rulemap['content_in'],rulemap['content_out'])
    convert_glm3(args.inputfile,args.outfile,rulemap['content_in'],rulemap['content_out'])
