//VERSION=3
function setup() {
    return {
      input: ["CO", "dataMask"],
      output: { bands: 4 }
    };
  }
  
  function evaluatePixel(sample) {
    // Map CO values to RGB color
    let CO = sample.CO;
    let color = getColor(CO);
  
    // If the data mask is 0, make the pixel fully transparent
    let alpha = sample.datamask ? 1 : 0;
  
    return [color[0], color[1], color[2], alpha];
  }
  
  function getColor(CO) {
    // Map CO values to a color gradient
    // Define a range of colors (from blue to red)
    let colors = [
      [0, 0, 255],    // Blue for low CO values
      [0, 255, 255],  // Cyan
      [0, 255, 0],    // Green
      [255, 255, 0],  // Yellow
      [255, 0, 0]     // Red for high CO values
    ];
  
    // Define the corresponding CO value range for the colors
    let COmin = 0.0; // Minimum CO value
    let COmax = 1.0; // Maximum CO value
    let range = COmax - COmin;
  
    // Normalize CO value to [0, 1]
    let normalizedCO = (CO - COmin) / range;
    normalizedCO = Math.max(0, Math.min(1, normalizedCO)); // Clamp to [0, 1]
  
    // Calculate the index and the fraction for interpolation
    let colorIndex = Math.floor(normalizedCO * (colors.length - 1));
    let color = colors[colorIndex];
  
    // Handle edge cases to avoid out-of-bounds access
    if (!color) {
      color = [0, 0, 0]; // Default to black if undefined
    }
  
    return color;
  }
  