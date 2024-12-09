import gradio as gr
import requests
from clearml import Task

# Initialize ClearML Task
task = Task.init(
    project_name="RAG System",  # Replace with your project name
    task_name="RAG Interface",
    task_type=Task.TaskTypes.inference  # You can change this based on your specific use case
)

# Define the FastAPI endpoint
API_URL = "http://localhost:8000/ask"

# Define pre-populated questions
QUESTIONS = [
    "What is ROS2?",
    "Tell me how can I navigate to a specific pose - include replanning aspects in your answer.",
    "Can you provide me with code for this task?"
    # You can add more questions here if needed
]

def ask_rag(question):
    """
    Sends the selected question to the FastAPI endpoint and retrieves the response.
    Parses the response to separate Q&A pairs and the final answer.

    Args:
        question (str): The selected question from the dropdown.

    Returns:
        tuple: A tuple containing the formatted Q&A pairs and the final answer.
    """
    # Log the input question
    task.get_logger().report_text(f"Question asked: {question}")

    payload = {"question": question}
    try:
        response = requests.post(API_URL, json=payload)
        if response.status_code == 200:
            answer = response.json().get("answer", "")
            
            # Initialize variables
            q_a_pairs = ""
            final_answer = ""
            
            # Define markers to split the response
            qa_marker = "Here are some relevant Q&A pairs:"
            answer_marker = "Now, answer the following question:"
            
            # Check if both markers are present in the response
            if qa_marker in answer and answer_marker in answer:
                # Split the response into Q&A pairs and final answer
                qa_part, final_part = answer.split(answer_marker, 1)
                
                # Clean and format Q&A pairs
                raw_q_a = qa_part.replace(qa_marker, "").strip()
                q_a_list = raw_q_a.split('\n\n')  # Assuming each Q&A pair is separated by double newlines
                
                formatted_q_a = []
                for pair in q_a_list:
                    if pair.startswith("Q:"):
                        q = pair.replace("Q:", "Q.").strip()
                        formatted_q_a.append(q)
                    elif pair.startswith("A:"):
                        a = pair.replace("A:", "A.").strip()
                        formatted_q_a.append(a)
                q_a_pairs = "\n\n".join(formatted_q_a)
                
                # Clean the final answer
                final_answer = final_part.strip()
            else:
                # If markers are not found, treat the entire answer as the final answer
                final_answer = answer.strip()
            
            # Log the results
            task.get_logger().report_text(f"Q&A Pairs:\n{q_a_pairs}")
            task.get_logger().report_text(f"Final Answer:\n{final_answer}")
            
            return q_a_pairs, final_answer
        else:
            # Handle non-200 responses
            error_detail = response.json().get('detail', 'Unknown error')
            error_message = f"Error: {error_detail}"
            
            # Log the error
            task.get_logger().report_error(error_message)
            
            return error_message, error_message
    except Exception as e:
        # Handle exceptions such as connection errors
        error_message = f"Error: {str(e)}"
        
        # Log the exception
        task.get_logger().report_error(error_message)
        
        return error_message, error_message

# Gradio interface setup
with gr.Blocks() as rag_app:
    # Title of the interface
    gr.Markdown("## RAG System Interface")
    
    # Dropdown for selecting a question
    with gr.Row():
        question_input = gr.Dropdown(
            choices=QUESTIONS,
            label="Select a Question",
            value=QUESTIONS[0]  # Set default value
        )
    
    # Button to submit the question
    submit_button = gr.Button("Ask")
    
    # Outputs: Q&A pairs and Final Answer
    with gr.Row():
        # Column for Q&A Pairs
        with gr.Column():
            gr.Markdown("### Relevant Q&A Pairs")
            q_a_output = gr.Markdown(value="")
        
        # Column for Final Answer
        with gr.Column():
            gr.Markdown("### Final Answer")
            answer_output = gr.Textbox(
                label="",
                value="",
                interactive=False,
                lines=10
            )
    
    # Define the action on button click
    submit_button.click(
        fn=ask_rag,
        inputs=question_input,
        outputs=[q_a_output, answer_output]
    )

# Launch the Gradio app
if __name__ == "__main__":
    rag_app.launch(server_name="0.0.0.0", server_port=7860)