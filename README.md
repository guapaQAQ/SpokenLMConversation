# SpokenLMConversation

This project involves running two different models (Model A and Model B) using configuration files and prompt datasets to generate conversations.

<!-- ## Requirements

Before running the script, ensure that you have the following dependencies installed:

- Python 3.x

Install any additional dependencies using `pip`:

```bash
pip install -r requirements.txt
``` -->

## How to Run

To run the conversation script with default configurations:

```bash
python offline_record.py --configA offline_record_config/config_cascade.yaml --configB offline_record_config/config_taipei1.yaml --prompt_path track2_chinese_prompt.json --output_dir ./conversation_outputs_cascade_spml-omni-step12864_onyx_temp0_rep2
```

## Argument Details

- `--configA`: The configuration file for Model A, which specifies model parameters and settings.
- `--configB`: The configuration file for Model B.
- `--prompt_path`: The path to the JSON file containing the prompts used in the conversation.
- `--output_dir`: The directory where conversation outputs will be saved.

## Example

```bash
python offline_record.py --configA configs/model_A.yaml --configB configs/model_B.yaml --prompt_path prompts/my_prompts.json --output_dir outputs/
```
