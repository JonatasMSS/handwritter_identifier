# Handwritten Letter and Digits Identification

This is a project that uses TensorFlow and Keras to classify handwritten texts, identifying whether the input is a number or a letter. The model is trained to recognize both handwritten digits and letters, allowing it to distinguish between them with high accuracy.

## Technologies Used
- **TensorFlow** and **Keras** for building and training the neural network model.
- **Python 3.11.9** for the development environment.
- **NumPy** for data manipulation.
- **Streamlit** for create a frontend interface 

## Installation Guide

Follow these steps to set up the project and run it on your local machine.

### 1. Clone the Repository

First, clone the repository to your local machine using the following command:

```bash
git clone https://github.com/JonatasMSS/handwritter_identifier.git
```

### 2. Set Up a Virtual Environment

It's recommended to use a virtual environment to manage dependencies. This ensures that the project uses the correct version of Python and all necessary libraries.

#### Create a Virtual Environment

Navigate to the project directory and create a virtual environment with Python 3.11.9:

```bash
python3.11 -m venv venv
```

#### Activate the Virtual Environment

- **Linux/macOS**:
  ```bash
  source venv/bin/activate
  ```

- **Windows**:
  ```bash
  venv\Scripts\activate
  ```

### 3. Install Dependencies

Once the virtual environment is activated, install the required dependencies listed in the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### 4. Running the Model

After installing the dependencies, you can run the model to classify handwritten letters and digits.

#### How to run

```bash
streamlit run main.py
```

The model will output whether the image contains a digit or a letter.

### 5. Deactivate the Virtual Environment

Once you're done working on the project, you can deactivate the virtual environment with the following command:

```bash
deactivate
```

## Contributing

Contributions to the project are welcome. If you have any improvements, bug fixes, or new features, feel free to fork the repository, create a new branch, and submit a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
