from llama_cpp import Llama
import gradio as gr
import argparse

# 定义参数
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str)
parser.add_argument('--max_tokens', type=int, default=128)
parser.add_argument('--temperature', type=float, default=0.5)
parser.add_argument('--top_p', type=float, default=0.5)
parser.add_argument('--top_k', type=int, default=40)
parser.add_argument('--repeat_penalty', type=float, default=1.5)
args = parser.parse_args()

# 加载模型
llm = Llama(args.model)

# 用户输入，清空输入框
def user_input(message:str, history:list):
    return "", history + [[message, None]]

def bot_predict(history:list):
    history[-1][1] = ''
    output = llm(prompt='\n'.join([f'Q: {x[0]}\nA: {x[1]}' for x in history]),
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                repeat_penalty=args.repeat_penalty,
                stop=["Q:"],
                stream=True)
    
    for each in output:
        history[-1][1] += each['choices'][0]['text']
        yield history


with gr.Blocks() as app:
    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Enter text and press enter")

    msg.submit(fn=user_input, inputs=[msg, chatbot], outputs=[msg, chatbot], ).then(
        fn=bot_predict, inputs=chatbot, outputs=chatbot)

app.queue().launch()