# How Obamify Works

Obamify is a revolutionary image transformation technology that rearranges pixels from a source image to recreate a target image (typically Barack Obama's face). The project combines sophisticated algorithms, GPU acceleration, and physics-based animation to create smooth transformations.

## Core Concept

At its heart, obamify solves an assignment problem: for each pixel in the target image, find the best matching pixel from the source image. The "best match" considers both color similarity and spatial proximity, weighted by user-configurable parameters.

## Technical Implementation

### 1. GPU-Accelerated Processing Pipeline

Obamify leverages modern graphics APIs (WebGPU/WebGL via wgpu) for high-performance computation:

#### Shader Programs
- **clear.wgsl**: Clears render targets to white
- **seed.wgsl**: Encodes seed (source pixel) positions and colors into textures
- **jfa.wgsl**: Implements Johnson's Find-All algorithm for computing Voronoi diagrams
- **shade.wgsl**: Final rendering step that assigns colors based on Voronoi cell assignments

#### Processing Flow
1. **Seed Preparation**: Source image pixels become "seeds" with positions and colors
2. **Target Preparation**: Target image pixels become target positions with associated colors
3. **Distance Calculation**: JFA computes for each target pixel which seed is closest
4. **Assignment Problem**: Solve the optimal assignment of seeds to targets
5. **Animation**: Animate seeds from source to target positions using physics simulation
6. **Rendering**: Draw final image where each pixel gets the color of its assigned target

### 2. Algorithms

#### Optimal Algorithm (Kuhn-Munkres/Hungarian Algorithm)
- Finds mathematically optimal solution minimizing total cost
- Cost function: `color_distance² + (proximity_weight × spatial_distance)²`
- Computationally expensive but guarantees optimal result
- Used for high-quality transformations

#### Genetic Algorithm
- Evolutionary approach that iteratively improves solution
- Representation: Permutation of source pixel indices
- Fitness: Negative total cost (lower cost = better)
- Operations: Swap pairs of pixels, accept if improves fitness
- Much faster but approximate solution
- Default algorithm for real-time interaction

### 3. Physics-Based Animation

Pixels are treated as particles with realistic motion:
- **Forces Acting on Particles**:
  - Destination Attraction: Pull toward target position
  - Wall Repulsion: Prevent particles from escaping boundaries
  - Neighbor Avoidance: Prevent overcrowding (personal space)
  - Stroke Cohesion: Keep related pixels together (for drawing mode)

- **Particle Properties**:
  - Position (current and target)
  - Velocity and acceleration
  - Age (for force modulation over time)
  - Stroke ID (for grouping related pixels)

### 4. WebAssembly & Web Support

- Compiles to WASM via `trunk` build system
- Uses Web Workers for background processing to maintain UI responsiveness
- Worker.js dynamically loads the WASM module
- Optimized for WebGL 2.0 compatibility with texture-based storage (instead of storage buffers)
- Progressive Web App (PWA) capabilities with service worker caching

### 5. User Interface

Built with egui (immediate mode GUI):
- Top panel: Controls for animation, preset selection, and transformation
- Settings dialog: Resolution, proximity importance, algorithm selection
- Real-time preview during configuration
- GIF recording functionality for sharing transformations
- Drawing mode (native only) for creating custom source images

## File Organization

```
src/
├── main.rs              # Application entry point
├── lib.rs               # Library exports
├── app.rs               # Core application logic
├── app/
│   ├── calculate/       # Processing algorithms
│   │   ├── mod.rs       # Algorithm selection and progress reporting
│   │   ├── util.rs      # Image processing, settings, progress types
│   │   ├── drawing_process.rs  # Native drawing functionality
│   │   └── worker/      # WASM worker interface
│   ├── gui.rs           # User interface implementation
│   ├── morph_sim.rs     # Particle physics simulation
│   ├── preset.rs        # Image preset data structures
│   └── app.rs           # App module
├── shaders/             # GPU shader programs (WGSL)
│   ├── clear.wgsl
│   ├── seed.wgsl
│   ├── jfa.wgsl
│   └── shade.wgsl
└── assets/              # Images, icons, preset data
```

## Key Data Structures

- **SeedPos**: 2D position of a particle (source pixel)
- **SeedColor**: RGBA color of a particle
- **ParamsCommon**: Uniform buffer with image dimensions and seed count
- **ParamsJfa**: Uniform buffer for JFA algorithm (step size)
- **Preset**: Processed image with assignments
- **UnprocessedPreset**: Raw image data waiting to be processed
- **GenerationSettings**: User-configurable algorithm parameters
- **ProgressMsg**: Communication between worker and main thread

## Performance Optimizations

1. **Texture-Based Storage**: Instead of GPU storage buffers (not widely supported in WebGL), uses textures to store seed/color data
2. **Batch Processing**: GPU operations process thousands of particles in parallel
3. **Incremental Updates**: Only changed data is uploaded to GPU each frame
4. **WebGL Compatibility**: Limits texture sizes and uses compatible formats
5. **Worker Offloading**: Heavy computations run in background threads/workers

## Usage Flow

1. **Image Selection**: Choose source and target images (or use presets)
2. **Configuration**: Adjust resolution, proximity importance, and algorithm
3. **Processing**: Algorithm runs to compute optimal pixel assignments
4. **Animation**: Particles animate from source to target positions
5. **Result**: Final transformed image is displayed
6. **Export**: Save as still image or animated GIF

## Mathematical Foundation

The core optimization problem minimizes:
```
Σ [w_c × Δcolor(i,j)² + w_s × Δposition(i,j)²]
```
Where:
- Δcolor(i,j) = Color distance between source pixel i and target pixel j
- Δposition(i,j) = Spatial distance between source pixel i and target pixel j
- w_c, w_s = Weights for color and spatial components (proximity_importance controls w_s)

This is solved as a linear sum assignment problem, which the Hungarian algorithm solves optimally in O(n³) time.
