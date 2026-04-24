const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const uploadContent = document.getElementById('upload-content');
const previewContainer = document.getElementById('preview-container');
const imagePreview = document.getElementById('image-preview');
const removeBtn = document.getElementById('remove-btn');
const processBtn = document.getElementById('process-btn');
const btnLoader = document.getElementById('btn-loader');
const btnText = document.querySelector('.btn-text');
const resultsContainer = document.getElementById('results-container');
const resultImage = document.getElementById('result-image');
const detectionStatus = document.getElementById('detection-status');
const countsGrid = document.getElementById('counts-grid');

let currentFile = null;

// Upload logic
dropZone.addEventListener('click', () => fileInput.click());

dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--primary)';
});

dropZone.addEventListener('dragleave', () => {
    dropZone.style.borderColor = 'var(--glass-border)';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'var(--glass-border)';
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        handleFile(file);
    }
});

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
});

function handleFile(file) {
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        uploadContent.style.display = 'none';
        previewContainer.style.display = 'block';
        processBtn.disabled = false;
        resultsContainer.style.display = 'none';
    };
    reader.readAsDataURL(file);
}

removeBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    currentFile = null;
    fileInput.value = '';
    uploadContent.style.display = 'block';
    previewContainer.style.display = 'none';
    processBtn.disabled = true;
    resultsContainer.style.display = 'none';
});

// Process logic
processBtn.addEventListener('click', async () => {
    if (!currentFile) return;

    // UI State
    processBtn.disabled = true;
    btnLoader.style.display = 'block';
    btnText.style.opacity = '0.5';

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.error) {
            alert(data.error);
            return;
        }

        displayResults(data);
    } catch (err) {
        console.error(err);
        alert('Failed to process image. Make sure the server is running and models exist.');
    } finally {
        processBtn.disabled = false;
        btnLoader.style.display = 'none';
        btnText.style.opacity = '1';
    }
});

function displayResults(data) {
    resultsContainer.style.display = 'block';
    resultsContainer.scrollIntoView({ behavior: 'smooth' });

    if (data.detected) {
        detectionStatus.textContent = 'Mangoes Detected';
        detectionStatus.style.background = 'rgba(76, 175, 80, 0.1)';
        detectionStatus.style.color = 'var(--secondary)';
        resultImage.src = `data:image/jpeg;base64,${data.image}`;
    } else {
        detectionStatus.textContent = 'No Mangoes Found';
        detectionStatus.style.background = 'rgba(239, 68, 68, 0.1)';
        detectionStatus.style.color = '#ef4444';
        resultImage.src = imagePreview.src;
    }

    countsGrid.innerHTML = '';
    for (const [label, count] of Object.entries(data.counts)) {
        const card = document.createElement('div');
        card.className = 'count-card';
        card.innerHTML = `
            <span class="label">${label}</span>
            <span class="value">${count}</span>
        `;
        countsGrid.appendChild(card);
    }
}
