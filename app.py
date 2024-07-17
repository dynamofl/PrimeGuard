import gradio as gr
from model import LitellmModel
from utils import route_templates, extract_and_eval_json
from jinja2 import Environment, FileSystemLoader
import json

# Load cached results
with open("cached_results.json", "r") as f:
    cached_results = json.load(f)

# Define constants
model = LitellmModel(model_id="mistral/open-mistral-7b")
env = Environment(loader=FileSystemLoader("templates"))
no_violation_template_name = "answer_utility.j2"
potential_violation_template_name_first_turn = "display_analysis.j2"
potential_violation_template_name_second_turn = "get_answer.j2"
direct_violation_template_name = "refusal.j2"


def process_input(prompt, cached_prompt):
    if cached_prompt:
        prompt = cached_prompt

    cached_result = next(
        (item for item in cached_results if item["prompt"] == prompt), None
    )

    if cached_result:
        route = cached_result["route"]
        vanilla_result = cached_result["vanilla_result"]
        primeguard_answer = cached_result["primeguard_result"]
        system_check_result = cached_result["system_check"]
        system_tip = cached_result["system_tip"]
        reevaluation = cached_result["reevaluation"]
    else:
        # Vanilla Approach
        system_instructions = env.get_template("oai_safety.j2").render()
        vanilla_result = model.batch_call([prompt], system_prompt=system_instructions)[
            0
        ]

        # PrimeGuard Approach
        routing_template = env.get_template("route_selection.j2")
        routing_rendered = routing_template.render(
            system_prompt=system_instructions, user_input=prompt
        )
        route_selection_output = model.batch_call([routing_rendered])[0]

        final_conv, routes, system_check_results, system_tips = route_templates(
            route_selection_outputs=[route_selection_output],
            prompts=[prompt],
            restrictive_system_instructions=system_instructions,
            env=env,
            no_violation_template_name=no_violation_template_name,
            potential_violation_template_name_first_turn=potential_violation_template_name_first_turn,
            potential_violation_template_name_second_turn=potential_violation_template_name_second_turn,
            direct_violation_template_name=direct_violation_template_name,
        )

        final_output = model.batch_call(final_conv)[0]

        primeguard_answer = final_output
        reevaluation = "N/A"
        if routes[0] == "potential_violation":
            parsed_json = extract_and_eval_json(final_output)
            if len(parsed_json) > 0:
                if (
                    "reevaluation" in parsed_json[0].keys()
                    and "final_response" in parsed_json[0].keys()
                ):
                    reevaluation = parsed_json[0]["reevaluation"]
                    primeguard_answer = parsed_json[0]["final_response"]

        route = routes[0]
        system_tip = system_tips[0]
        system_check_result = system_check_results[0]

    button_updates = [
        gr.update(variant="secondary"),
        gr.update(variant="secondary"),
        gr.update(variant="secondary"),
    ]
    if route == "no_to_minimal_risk":
        button_updates[0] = gr.update(variant="primary")
    elif route == "potential_violation":
        button_updates[1] = gr.update(variant="primary")
    elif route == "direct_violation":
        button_updates[2] = gr.update(variant="primary")

    return (
        vanilla_result,
        primeguard_answer,
        *button_updates,
        system_check_result,
        system_tip,
        reevaluation,
        prompt,  # Return the prompt to update the input field
    )


css = """
.route-button { height: 50px; }
"""

with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
    gr.Markdown("# 🤺 PrimeGuard Demo 🤺")

    with gr.Row():
        with gr.Column(scale=3):
            prompt_input = gr.Textbox(
                label="Enter your prompt", lines=3, placeholder="You can't break me"
            )
            submit_btn = gr.Button("Submit", variant="primary")

        with gr.Column(scale=1):
            gr.Markdown("### Cached Examples")
            cached_prompts = gr.Dropdown(
                choices=[item["prompt"] for item in cached_results],
                label="Select a cached prompt",
                allow_custom_value=True,
            )

    with gr.Row():
        vanilla_output = gr.Textbox(
            label="Mistral 7B Defended with System Prompt 🤡",
            lines=5,
            interactive=False,
        )
        primeguard_output = gr.Textbox(
            label="Mistral 7B Defended with 🤺 PrimeGuard 🤺",
            lines=5,
            interactive=False,
        )

    gr.Markdown("## PrimeGuard Details")

    with gr.Row():
        no_risk = gr.Button(
            "No to Minimal Risk", variant="secondary", elem_classes=["route-button"]
        )
        potential_violation = gr.Button(
            "Potential Violation", variant="secondary", elem_classes=["route-button"]
        )
        direct_violation = gr.Button(
            "Direct Violation", variant="secondary", elem_classes=["route-button"]
        )

    with gr.Column():
        system_check = gr.Textbox(
            label="System Check Result", lines=3, interactive=False
        )
        system_tip = gr.Textbox(label="System Tip", lines=3, interactive=False)
        reevaluation = gr.Textbox(label="Reevaluation", lines=3, interactive=False)

    with gr.Row():
        gr.HTML(
            """<a href="https://www.dynamofl.com" target="_blank">
                    <p align="center">
                        <img src="https://bookface-images.s3.amazonaws.com/logos/4decc4e1a1e133a40d326cb8339c3a52fcbfc4dc.png" alt="Dynamo" width="200">
                    </p>
                </a>
        """,
            elem_id="ctr",
        )

    def update_ui(
        vanilla_result,
        primeguard_result,
        no_risk_update,
        potential_violation_update,
        direct_violation_update,
        system_check_result,
        system_tip,
        reevaluation,
        prompt,
    ):
        return [
            vanilla_result,
            primeguard_result,
            no_risk_update,
            potential_violation_update,
            direct_violation_update,
            system_check_result,
            system_tip,
            reevaluation,
            prompt,  # Update the prompt input field
        ]

    def reset_cached_prompt():
        return gr.update(value=None)

    submit_btn.click(
        fn=process_input,
        inputs=[prompt_input, cached_prompts],
        outputs=[
            vanilla_output,
            primeguard_output,
            no_risk,
            potential_violation,
            direct_violation,
            system_check,
            system_tip,
            reevaluation,
            prompt_input,
        ],
    ).then(
        fn=update_ui,
        inputs=[
            vanilla_output,
            primeguard_output,
            no_risk,
            potential_violation,
            direct_violation,
            system_check,
            system_tip,
            reevaluation,
            prompt_input,
        ],
        outputs=[
            vanilla_output,
            primeguard_output,
            no_risk,
            potential_violation,
            direct_violation,
            system_check,
            system_tip,
            reevaluation,
            prompt_input,
        ],
    ).then(
        fn=reset_cached_prompt,
        inputs=[],
        outputs=[cached_prompts],
    )

    # Add an event listener for the cached_prompts dropdown
    cached_prompts.change(
        fn=process_input,
        inputs=[prompt_input, cached_prompts],
        outputs=[
            vanilla_output,
            primeguard_output,
            no_risk,
            potential_violation,
            direct_violation,
            system_check,
            system_tip,
            reevaluation,
            prompt_input,
        ],
    ).then(
        fn=update_ui,
        inputs=[
            vanilla_output,
            primeguard_output,
            no_risk,
            potential_violation,
            direct_violation,
            system_check,
            system_tip,
            reevaluation,
            prompt_input,
        ],
        outputs=[
            vanilla_output,
            primeguard_output,
            no_risk,
            potential_violation,
            direct_violation,
            system_check,
            system_tip,
            reevaluation,
            prompt_input,
        ],
    )

demo.queue(max_size=20)
demo.launch(max_threads=40)
