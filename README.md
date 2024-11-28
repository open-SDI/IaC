# IaC Generation and Execution with LLM

This project is designed to generate and execute Ansible playbooks using Large Language Models (LLMs). It takes a user requirement, refines it into a prompt, and generates a playbook through multiple iterations of refining and validation.

## Features

- Refine user requirements using an LLM
- Generate end-state descriptions in JSON format
- Concurrently request IaC code generation from multiple LLMs
- Validate and format the generated playbook
- Execute the Ansible playbook and capture errors
- Iteratively refine the playbook based on execution feedback

## Prerequisites

- Python 3.7+
- Ansible

## Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/open-SDI/IaC.git
    cd IaC
    ```

2. Create a virtual environment and activate it:

    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required Python packages:

    ```sh
    pip install -r requirements.txt
    ```

4. Ensure Ansible is installed on your system. You can install it using:

    ```sh
    pip install ansible
    ```

## Usage

1. Run the main script:

    ```sh
    python main.py
    ```

2. Enter your requirement for the Ansible playbook when prompted.

3. Provide additional information if requested.

4. The script will generate and execute the playbook iteratively, refining it based on the feedback and errors encountered.

## Configuration

- The configuration for LLM endpoints and models is defined in the script.
- You may need to update the IP addresses and model IDs as per your setup.

## Logging

- Logs are saved in the `playbook_output_<timestamp>` directory created during execution.
- Logs include detailed information about each iteration, including prompts, responses, and errors.

## Acknowledgements

- [Requests](https://docs.python-requests.org/)
- [PyYAML](https://pyyaml.org/)
- [Ansible](https://www.ansible.com/)


