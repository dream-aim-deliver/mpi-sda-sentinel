//VERSION=3
var minVal = 0.0;
var maxVal = 0.1;
var diff = maxVal - minVal;
const map = [
    [minVal, 0x00007f], 
    [minVal + 0.125 * diff, 0x0000ff],
    [minVal + 0.375 * diff, 0x00ffff],
    [minVal + 0.625 * diff, 0xffff00],
    [minVal + 0.875 * diff, 0xff0000],
    [maxVal, 0x7f0000]
]; 

const visualizer = new ColorRampVisualizer(map);

function setup() {
    return {
        input: ["CO", "dataMask"],
        output: { bands: 4 }
    };
}

function evaluatePixel(samples) {
    const [r, g, b] = visualizer.process(samples.CO);
    return [r, g, b, samples.dataMask];
}

// Helper function to get the dominant color in a grid cell
function getDominantColor(colors) {
    const colorCounts = {};
    let maxCount = 0;
    let dominantColor = null;
    
    colors.forEach(color => {
        const key = color.join(',');
        colorCounts[key] = (colorCounts[key] || 0) + 1;
        
        if (colorCounts[key] > maxCount) {
            maxCount = colorCounts[key];
            dominantColor = color;
        }
    });
    
    return dominantColor;
}

// Main function to process the image into a grid
function processImageIntoGrid(imageData, gridSize) {
    const width = imageData.width;
    const height = imageData.height;
    const numGridX = Math.ceil(width / gridSize);
    const numGridY = Math.ceil(height / gridSize);
    const gridResults = [];

    for (let gx = 0; gx < numGridX; gx++) {
        for (let gy = 0; gy < numGridY; gy++) {
            const colors = [];
            for (let x = gx * gridSize; x < (gx + 1) * gridSize && x < width; x++) {
                for (let y = gy * gridSize; y < (gy + 1) * gridSize && y < height; y++) {
                    const index = (y * width + x) * 4;
                    const r = imageData.data[index];
                    const g = imageData.data[index + 1];
                    const b = imageData.data[index + 2];
                    const alpha = imageData.data[index + 3];
                    
                    if (alpha > 0) { // Consider only non-transparent pixels
                        colors.push([r, g, b]);
                    }
                }
            }

            if (colors.length > 0) {
                const dominantColor = getDominantColor(colors);
                const concentration = getConcentrationFromColor(dominantColor);
                gridResults.push({
                    x: gx,
                    y: gy,
                    concentration: concentration
                });
            }
        }
    }

    return gridResults;
}

// Helper function to map RGB color to concentration
function getConcentrationFromColor(color) {
    const [r, g, b] = color;
    const hexColor = (r << 16) | (g << 8) | b;

    for (let i = 0; i < map.length; i++) {
        if (map[i][1] === hexColor) {
            return map[i][0];
        }
    }

    return null;
}
