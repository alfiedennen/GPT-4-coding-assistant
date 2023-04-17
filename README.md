# GPT-4 Coding Assistant

GPT-4 Coding Assistant is a web application that leverages the power of OpenAI's GPT-4 to help developers with their coding tasks. The application serves as an interactive chatbot that assists in code generation, understanding, and troubleshooting. It also utilizes embeddings and the Annoy library to search for similar code snippets in the provided codebase, offering more contextually relevant responses.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)

## Features

- Interact with GPT-4 to generate and understand code snippets
- Leverage embeddings and the Annoy library to search for similar code snippets in the provided codebase
- Utilize the web interface to communicate with the GPT-4 model
- Track token usage during the conversation
- Enable or disable embeddings lookup during the conversation

## Installation

1. Clone the repository:

```
git clone https://github.com/alfiedennen/gpt-4-coding-assistant.git
```

2. Change to the project directory:

```
cd gpt-4-coding-assistant
```

3. Install the required dependencies:

```
pip install -r requirements.txt
```

4. Create embeddings for your codebase:

```
python create_embeddings.py
```

## Usage

1. Run the Flask web application:

```
python app.py
```

2. Visit the web interface at http://localhost:5000.

3. Start interacting with the GPT-4 Coding Assistant by typing your code-related queries or requests in the chat interface.

4. Enable or disable the embeddings lookup by toggling the checkbox.

## Customization

You can customize the GPT-4 Coding Assistant to work with your codebase by following these steps:

1. Replace the contents of the `code` directory with your codebase.

2. Re-run the `create_embeddings.py` script to generate embeddings for your codebase:

```
python create_embeddings.py
```

## Contributing

We welcome contributions to improve the GPT-4 Coding Assistant. To contribute, follow these steps:

1. Fork the repository.

2. Create a new branch with a descriptive name:

```
git checkout -b my-feature-branch
```

3. Implement your changes or improvements.

4. Commit the changes and create a pull request.

5. Wait for the maintainers to review and merge the changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
