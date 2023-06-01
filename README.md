# QueryGPT

This repository contains a Python application that uses the Langchain library to optimize database queries. The application reads a list of queries from a CSV file, sends each query to the Langchain library for optimization, and then writes the optimized queries and any notes about the optimization process back to a CSV file.

## Getting Started

### Prerequisites
- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Clone the repository
    ```sh
    git clone https://github.com/alexnodeland/querygpt.git
    cd QueryGPT
    ```
2. Install the required packages
    ```sh
    pip install -r requirements.txt
    ```
3. Set your OpenAI API Key as an environment variable
    ```sh
    export OPENAI_API_KEY=<your-api-key>
    ```
    Replace your-api-key with your actual OpenAI API key.

## Usage

1. Update the config.py file with your configuration settings.
2. Run the application
    ```sh
    python main.py
    ```
    The script will read the queries from the CSV file specified in config.py, optimize the queries using Langchain, and write the optimized queries and notes to the output CSV file specified in config.py.

## Contributing

I welcome contributions to this project. Please feel free to submit a pull request or open an issue on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

If you have any questions or feedback, please open an issue on GitHub.

## Acknowledgements

This project uses the Langchain library to optimize SQL queries.
