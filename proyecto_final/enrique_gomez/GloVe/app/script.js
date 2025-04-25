document.addEventListener('DOMContentLoaded', () => {
    // --- DOM Elements ---
    const fileInput = document.getElementById('file-input');
    const toggleDimButton = document.getElementById('toggle-dim');
    const currentDimSpan = document.getElementById('current-dim');
    const arithmeticInput = document.getElementById('arithmetic-input');
    const calculateButton = document.getElementById('calculate-arithmetic');
    const resultDisplay = document.getElementById('arithmetic-result');
    const statusDiv = document.getElementById('status');
    const plotDiv = document.getElementById('plot');

    // --- State Variables ---
    let embeddings = {}; // { word: [num, num, ...], ... }
    let words = [];      // [word1, word2, ...] (maintains order)
    let vectors = [];    // [[num, num, ...], ...] (parallel to words)
    let tsneModel = null;
    let plotData2D = null; // { x: [], y: [], text: [] }
    let plotData3D = null; // { x: [], y: [], z: [], text: [] }
    let currentDim = 2;
    let highlightedIndices = []; // Indices of words to highlight
    let isLoading = false;

    // --- Initialization ---
    updateButtonStates();
    fileInput.addEventListener('change', handleFileSelect);
    toggleDimButton.addEventListener('click', toggleDimension);
    calculateButton.addEventListener('click', handleArithmetic);

    // --- Functions ---

    function updateStatus(message, append = false) {
        console.log(message); // Log to console as well
        if (append) {
            statusDiv.innerHTML += `<br>${message}`;
        } else {
            statusDiv.textContent = message;
        }
        // Scroll status to bottom
        statusDiv.scrollTop = statusDiv.scrollHeight;
    }

    function updateButtonStates() {
        const embeddingsLoaded = words.length > 0;
        toggleDimButton.disabled = !plotData2D || isLoading; // Disable if no data or loading
        calculateButton.disabled = !embeddingsLoaded || isLoading;
        arithmeticInput.disabled = !embeddingsLoaded || isLoading;
        fileInput.disabled = isLoading;

        if (isLoading) {
             updateStatus("Processing...");
             if (toggleDimButton.textContent.startsWith("Switch")) {
                 toggleDimButton.textContent = "Processing...";
             }
             if (calculateButton.textContent.startsWith("Calculate")) {
                 calculateButton.textContent = "Processing...";
             }
        } else {
             toggleDimButton.textContent = `Switch to ${currentDim === 2 ? '3D' : '2D'} Plot`;
             calculateButton.textContent = "Calculate & Highlight";
        }
    }

    async function handleFileSelect(event) {
        const file = event.target.files[0];
        if (!file) return;

        resetState();
        isLoading = true;
        updateButtonStates();
        updateStatus(`Loading file: ${file.name}...`);

        const reader = new FileReader();
        reader.onload = async (e) => {
            try {
                updateStatus('Parsing embeddings...');
                await parseEmbeddings(e.target.result);
                updateStatus(`Loaded ${words.length} words with dimension ${vectors[0]?.length || 'N/A'}.`);
                if (words.length > 0) {
                    updateStatus('Running t-SNE (this may take a while)...');
                    // Use setTimeout to allow UI update before heavy computation
                    setTimeout(async () => {
                        await runTSNE();
                        isLoading = false;
                        updateButtonStates();
                        updateStatus('t-SNE complete. Plotting...');
                        plotEmbeddings();
                    }, 50); // Small delay
                } else {
                     isLoading = false;
                     updateButtonStates();
                     updateStatus('No valid embeddings found in the file.');
                }
            } catch (error) {
                isLoading = false;
                updateButtonStates();
                updateStatus(`Error processing file: ${error.message}`);
                console.error(error);
                resetState(); // Clear potentially partial data
            }
        };
        reader.onerror = () => {
            isLoading = false;
            updateButtonStates();
            updateStatus(`Error reading file: ${reader.error}`);
            resetState();
        };
        reader.readAsText(file);
    }

     function resetState() {
        embeddings = {};
        words = [];
        vectors = [];
        tsneModel = null;
        plotData2D = null;
        plotData3D = null;
        highlightedIndices = [];
        currentDim = 2; // Reset to 2D
        currentDimSpan.textContent = `Current: 2D`;
        resultDisplay.textContent = 'N/A';
        arithmeticInput.value = '';
        Plotly.purge(plotDiv); // Clear previous plot
        updateStatus('Ready. Load an embedding file.');
    }

    async function parseEmbeddings(fileContent) {
        return new Promise((resolve, reject) => {
            try {
                const lines = fileContent.trim().split('\n');
                let tempEmbeddings = {};
                let tempWords = [];
                let tempVectors = [];
                let firstVectorLength = -1;

                lines.forEach((line, index) => {
                    const parts = line.trim().split(' ');
                    if (parts.length < 2) {
                        console.warn(`Skipping malformed line ${index + 1}: ${line}`);
                        return; // Skip lines that don't have at least a word and one dim
                    }

                    const word = parts[0];
                    const vec = parts.slice(1).map(Number);

                    // Check for NaN values
                    if (vec.some(isNaN)) {
                        console.warn(`Skipping line ${index + 1} due to non-numeric vector data: ${word}`);
                        return;
                    }

                    // Check for consistent vector length
                    if (firstVectorLength === -1) {
                        firstVectorLength = vec.length;
                    } else if (vec.length !== firstVectorLength) {
                         console.warn(`Skipping line ${index + 1} due to inconsistent vector length: ${word} (expected ${firstVectorLength}, got ${vec.length})`);
                         return;
                    }

                    if (word && vec.length > 0) {
                        // Handle potential duplicate words (keep the first occurrence)
                        if (!tempEmbeddings[word]) {
                            tempEmbeddings[word] = vec;
                            tempWords.push(word);
                            tempVectors.push(vec);
                        } else {
                             console.warn(`Duplicate word found: "${word}". Keeping first occurrence.`);
                        }
                    }
                });

                if (tempWords.length === 0) {
                    reject(new Error("No valid word embeddings found in the file. Check format (word val1 val2 ...)."));
                    return;
                }

                embeddings = tempEmbeddings;
                words = tempWords;
                vectors = tempVectors;
                resolve();
            } catch (error) {
                reject(new Error(`Parsing failed: ${error.message}`));
            }
        });
    }

    async function runTSNE() {
        return new Promise(async (resolve, reject) => {
            if (vectors.length === 0) {
                reject(new Error("No vectors loaded to run t-SNE."));
                return;
            }

            // --- t-SNE for 2D ---
            updateStatus('Running t-SNE for 2D...', true);
            try {
                const opt2D = {
                    epsilon: 10, // Learning rate
                    perplexity: Math.min(30, Math.floor(vectors.length / 3) -1), // Typical value, adjust based on dataset size
                    dim: 2      // Output dimensions
                };
                 if (opt2D.perplexity <= 0) opt2D.perplexity = 1; // Handle very small datasets

                tsneModel = new tsnejs.tSNE(opt2D);
                tsneModel.initDataRaw(vectors);

                // Run t-SNE iterations
                // Run more iterations for potentially better results, but takes longer
                const maxIter = 500;
                for (let k = 0; k < maxIter; k++) {
                    tsneModel.step();
                    if (k % 50 === 0) {
                         updateStatus(`t-SNE 2D iteration ${k}/${maxIter}...`, true);
                         // Yield to event loop to keep UI responsive
                         await new Promise(requestAnimationFrame);
                    }
                }

                const output2D = tsneModel.getSolution(); // Get final solution
                plotData2D = {
                    x: output2D.map(v => v[0]),
                    y: output2D.map(v => v[1]),
                    text: words // Hover text
                };
                updateStatus('2D t-SNE complete.', true);

            } catch(error) {
                 updateStatus(`2D t-SNE failed: ${error.message}`, true);
                 console.error("2D t-SNE Error:", error);
                 // Don't reject the whole process, maybe 3D will work or user wants 2D only
            }


            // --- t-SNE for 3D ---
             updateStatus('Running t-SNE for 3D...', true);
             try {
                const opt3D = {
                    epsilon: 10,
                    perplexity: Math.min(30, Math.floor(vectors.length / 3) -1),
                    dim: 3
                };
                 if (opt3D.perplexity <= 0) opt3D.perplexity = 1;

                // Re-initialize or create a new model for 3D
                tsneModel = new tsnejs.tSNE(opt3D); // Use new instance if params differ significantly or model state is complex
                tsneModel.initDataRaw(vectors);

                const maxIter3D = 500; // Can use same or different iterations
                for (let k = 0; k < maxIter3D; k++) {
                    tsneModel.step();
                     if (k % 50 === 0) {
                         updateStatus(`t-SNE 3D iteration ${k}/${maxIter3D}...`, true);
                         await new Promise(requestAnimationFrame);
                    }
                }

                const output3D = tsneModel.getSolution();
                plotData3D = {
                    x: output3D.map(v => v[0]),
                    y: output3D.map(v => v[1]),
                    z: output3D.map(v => v[2]),
                    text: words
                };
                 updateStatus('3D t-SNE complete.', true);

             } catch(error) {
                 updateStatus(`3D t-SNE failed: ${error.message}`, true);
                 console.error("3D t-SNE Error:", error);
             }

            resolve(); // Resolve even if one dimension failed, maybe the other worked
        });
    }

    function plotEmbeddings() {
        if ((currentDim === 2 && !plotData2D) || (currentDim === 3 && !plotData3D)) {
            updateStatus(`Plot data for ${currentDim}D not available.`);
            Plotly.purge(plotDiv); // Clear the plot area
            return;
        }

        const plotData = currentDim === 2 ? plotData2D : plotData3D;
        if (!plotData) return; // Should not happen based on above check, but safety first

        const traceBase = {
            x: plotData.x,
            y: plotData.y,
            text: plotData.text,
            mode: 'markers',
            type: currentDim === 2 ? 'scatter' : 'scatter3d',
            hoverinfo: 'text',
            marker: {
                size: 5,
                opacity: 0.7,
                // Color gradient - using index as a proxy for some order/grouping
                // A more sophisticated approach might color based on distance to a selected point
                // or cluster ID if clustering was performed. t-SNE itself provides the grouping visually.
                color: plotData.x.map((_, i) => i), // Color by index
                colorscale: 'Viridis', // Example colorscale
                // colorbar: { title: 'Word Index' } // Optional color bar
            }
        };

        if (currentDim === 3) {
            traceBase.z = plotData.z;
        }

        // --- Highlighting Logic ---
        let traces = [];
        if (highlightedIndices.length > 0) {
            const highlightTrace = {
                x: [], y: [], text: [],
                mode: 'markers',
                type: traceBase.type,
                hoverinfo: 'text',
                marker: {
                    size: 10, // Larger size
                    color: 'red', // Distinct color
                    opacity: 1.0,
                    symbol: 'diamond' // Different symbol
                },
                name: 'Highlighted' // Legend entry
            };
            if (currentDim === 3) highlightTrace.z = [];

            const nonHighlightTrace = {
                x: [], y: [], text: [],
                mode: 'markers',
                type: traceBase.type,
                hoverinfo: 'text',
                marker: {
                    size: 5,
                    opacity: 0.6,
                    color: plotData.x.map((_, i) => i), // Keep original coloring for non-highlighted
                    colorscale: 'Viridis',
                },
                 name: 'Embeddings' // Legend entry
            };
             if (currentDim === 3) nonHighlightTrace.z = [];

            const highlightedSet = new Set(highlightedIndices);

            for (let i = 0; i < words.length; i++) {
                if (highlightedSet.has(i)) {
                    highlightTrace.x.push(plotData.x[i]);
                    highlightTrace.y.push(plotData.y[i]);
                    if (currentDim === 3) highlightTrace.z.push(plotData.z[i]);
                    highlightTrace.text.push(plotData.text[i]);
                } else {
                    nonHighlightTrace.x.push(plotData.x[i]);
                    nonHighlightTrace.y.push(plotData.y[i]);
                    if (currentDim === 3) nonHighlightTrace.z.push(plotData.z[i]);
                    nonHighlightTrace.text.push(plotData.text[i]);
                }
            }
            traces.push(nonHighlightTrace); // Draw non-highlighted first
            traces.push(highlightTrace);    // Draw highlighted on top
        } else {
            // No highlighting, just the base trace
            traceBase.name = 'Embeddings';
            traces.push(traceBase);
        }


        const layout = {
            title: `Word Embeddings (${currentDim}D t-SNE)`,
            margin: { l: 0, r: 0, b: 0, t: 40 }, // Adjust margins
            hovermode: 'closest',
             scene: currentDim === 3 ? { // Specific layout for 3D
                xaxis: { title: 'TSNE-1' },
                yaxis: { title: 'TSNE-2' },
                zaxis: { title: 'TSNE-3' },
            } : { // Specific layout for 2D
                 xaxis: { title: 'TSNE-1' },
                 yaxis: { title: 'TSNE-2' },
            },
            showlegend: highlightedIndices.length > 0 // Show legend only when highlighting
        };

        Plotly.react(plotDiv, traces, layout); // Use react for efficient updates
    }

    function toggleDimension() {
        if (isLoading) return;
        currentDim = currentDim === 2 ? 3 : 2;
        currentDimSpan.textContent = `Current: ${currentDim}D`;
        toggleDimButton.textContent = `Switch to ${currentDim === 2 ? '3D' : '2D'} Plot`;

        // Check if data for the new dimension exists
        if ((currentDim === 3 && !plotData3D) || (currentDim === 2 && !plotData2D)) {
             updateStatus(`Plot data for ${currentDim}D is not available. Recalculating might be needed if t-SNE failed.`);
             // Optionally, trigger runTSNE again here if needed, but for now, just show message
             Plotly.purge(plotDiv); // Clear plot if data is missing
        } else {
            plotEmbeddings(); // Re-plot with the new dimension's data
        }
    }

    // --- Vector Arithmetic Functions ---

    function vecAdd(vecA, vecB) {
        if (!vecA || !vecB || vecA.length !== vecB.length) return null;
        return vecA.map((val, i) => val + vecB[i]);
    }

    function vecSubtract(vecA, vecB) {
         if (!vecA || !vecB || vecA.length !== vecB.length) return null;
        return vecA.map((val, i) => val - vecB[i]);
    }

    function vecMagnitude(vec) {
        if (!vec) return 0;
        return Math.sqrt(vec.reduce((sum, val) => sum + val * val, 0));
    }

    function cosineSimilarity(vecA, vecB) {
        if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
        const dotProduct = vecA.reduce((sum, val, i) => sum + val * vecB[i], 0);
        const magA = vecMagnitude(vecA);
        const magB = vecMagnitude(vecB);
        if (magA === 0 || magB === 0) return 0; // Handle zero vectors
        return dotProduct / (magA * magB);
    }

    function findNearestNeighbor(targetVec, excludeIndices = new Set()) {
        let bestWordIndex = -1;
        let maxSimilarity = -Infinity; // Cosine similarity ranges from -1 to 1

        for (let i = 0; i < vectors.length; i++) {
            if (excludeIndices.has(i)) continue; // Skip excluded words

            const sim = cosineSimilarity(targetVec, vectors[i]);
            if (sim > maxSimilarity) {
                maxSimilarity = sim;
                bestWordIndex = i;
            }
        }
        return bestWordIndex; // Return the index
    }

    function handleArithmetic() {
        if (isLoading) return;
        const expression = arithmeticInput.value.trim().toLowerCase();
        if (!expression) {
            resultDisplay.textContent = 'N/A';
            highlightedIndices = [];
            plotEmbeddings(); // Redraw without highlights
            return;
        }

        const parts = expression.split(/\s+/); // Split by whitespace
        let currentVector = null;
        let operation = '+'; // Implicit '+' for the first word
        let error = null;
        let involvedWordIndices = new Set(); // Use a Set for efficient lookup

        updateStatus(`Calculating: ${expression}`);

        for (let i = 0; i < parts.length; i++) {
            const part = parts[i];

            if (part === '+' || part === '-') {
                operation = part;
            } else {
                // It's a word
                const word = part;
                const wordIndex = words.indexOf(word);

                if (wordIndex === -1) {
                    error = `Word not found: "${word}"`;
                    break;
                }

                const vector = embeddings[word];
                involvedWordIndices.add(wordIndex); // Add index of word used in calculation

                if (!currentVector) {
                    // First word
                    currentVector = [...vector]; // Make a copy
                } else {
                    // Subsequent words
                    if (operation === '+') {
                        currentVector = vecAdd(currentVector, vector);
                    } else { // operation === '-'
                        currentVector = vecSubtract(currentVector, vector);
                    }

                    if (!currentVector) { // Check if add/subtract failed (shouldn't if parsing is correct)
                         error = "Vector operation failed (dimension mismatch?).";
                         break;
                    }
                }
            }
        }

        if (error) {
            updateStatus(`Arithmetic Error: ${error}`);
            resultDisplay.innerHTML = `<span style="color: red;">Error</span>`;
            highlightedIndices = []; // Clear highlights on error
        } else if (currentVector) {
            // Find the nearest neighbor to the resulting vector
            const nearestIndex = findNearestNeighbor(currentVector, involvedWordIndices);

            if (nearestIndex !== -1) {
                const resultWord = words[nearestIndex];
                resultDisplay.innerHTML = `<strong>${resultWord}</strong>`;
                updateStatus(`Result: ${resultWord}`);
                // Highlight input words AND the result word
                highlightedIndices = [...involvedWordIndices, nearestIndex];
            } else {
                resultDisplay.textContent = 'Could not find neighbor';
                 updateStatus('Could not find a nearest neighbor.');
                highlightedIndices = [...involvedWordIndices]; // Highlight only inputs if no result found
            }
        } else {
             resultDisplay.textContent = 'Invalid expression';
             updateStatus('Invalid arithmetic expression.');
             highlightedIndices = [];
        }

        plotEmbeddings(); // Update plot with new highlights
    }
});
