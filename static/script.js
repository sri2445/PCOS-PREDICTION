
document.addEventListener('DOMContentLoaded', function() {
    const welcomeContainer = document.getElementById('welcomeContainer');
    const formContainer = document.getElementById('formContainer');
    const startButton = document.getElementById('startButton');
    const predictionForm = document.getElementById('predictionForm');
    const predictionResultDisplay = document.getElementById('predictionResultDisplay');
    const resetButton = document.getElementById('resetButton');

    startButton.addEventListener('click', function() {
        welcomeContainer.style.display = 'none';
        formContainer.style.display = 'block';
        predictionResultDisplay.textContent = ''; // Clear previous result
        resetButton.style.display = 'none';
    });

    resetButton.addEventListener('click', function() {
        welcomeContainer.style.display = 'block';
        formContainer.style.display = 'none';
        predictionForm.reset(); // Clear form data
        predictionResultDisplay.textContent = '';
        resetButton.style.display = 'none';
    });

    predictionForm.addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent default form submission

        const formData = new FormData(predictionForm);

        fetch('/', {
            method: 'POST',
            body: formData
        })
        .then(response => response.text())
        .then(data => {
            // Log the response to help debug
            console.log('Server Response:', data);

            // Parse the response and get the prediction result
            const resultDiv = document.createElement('div');
            resultDiv.innerHTML = data; // Adding the response HTML to a new div
            
            // Find the prediction result from the returned HTML
            const predictionElement = resultDiv.querySelector('#result');
            if (predictionElement) {
                predictionResultDisplay.textContent = predictionElement.textContent;
            } else {
                predictionResultDisplay.textContent = 'Error: Could not retrieve prediction.';
            }

            formContainer.style.display = 'none';
            welcomeContainer.style.display = 'block';
            resetButton.style.display = 'block';
        })
        .catch(error => {
            console.error('Error submitting form:', error);
            predictionResultDisplay.textContent = 'Error: Could not connect to the server.';
            formContainer.style.display = 'none';
            welcomeContainer.style.display = 'block';
            resetButton.style.display = 'block';
        });
    });
});