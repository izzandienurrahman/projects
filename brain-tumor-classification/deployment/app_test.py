import gradio as gr

def greet(name):
    return "Hello " + name + "!"
img_inputs = gr.inputs.Image(shape=(256, 256, 3), label="Upload Image:")
txt_input = gr.Textbox(lines=2, placeholder="Name Here...")
demo = gr.Interface(
    fn=greet,
    inputs=img_inputs,
    outputs="text",
)
demo.launch()