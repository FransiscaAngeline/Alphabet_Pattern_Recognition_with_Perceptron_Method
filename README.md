# Alphabet PAttern Recognition with Perceptron Method
## D4 JTD 3A Group 1 
- Fakhri Khoiruzaki (2241160060)
- Fransisca Angeline Susanto (2241160094)
- Qoulan Nurza Sadiida Enmala (2241160054)

## Description 
This project uses the **Perceptron** method to recognize alphabet letter patterns drawn by users on a 5x5 grid. This project is built using HTML for the frontend, and Flask to manage the backend and handle the pattern recognition endpoints.

## Features

- **5x5 grid**: Users can draw alphabet letter patterns by clicking the boxes on the 5x5 grid.
- **Pattern Recognition**: Once the pattern is drawn, the system will identify the entered letters using the Perceptron algorithm.
- Flask Backend**: Flask serves to handle pattern recognition requests and provide prediction results.
- Interactivity**: Users can redraw patterns and try recognition for different letters.

## Technology Used

- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask (Python)
- **Machine Learning**: Perceptron Algorithm
- **Library**: Flask (for web framework),NumPy (for numerical computation), JSON (for data storage and reading), OS (for file system operation)


## Perceptron model
The Perceptron model is implemented in the app.py file. The pattern has been trained with 5x5 letter pattern data to recognize the A-Z alphabet. The training process can be re-executed if required, and the training code is included in this file.

## Training and Testing  
- Training: Try entering some letter patterns and check if the system recognizes the letters correctly.
- Testing: Users can input letter patterns according to those in the database that has been created or input different letter patterns to test whether the system recognizes letters or not. 
