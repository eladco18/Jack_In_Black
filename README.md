# JACK_in_BLACK 🃏

**Advanced Real-Time Computer Vision System for Blackjack Analytics and Strategy Recommendation.**

[ **Insert GIF Demo of the system in action here** ]

JACK_in_BLACK is a final project developed for the "Introduction to Image Processing" course. The system tracks a physical Blackjack table in real-time, detecting cards and casino chips, and provides live statistical analysis and strategy recommendations.
The core challenge and unique achievement of this project is that **all complex object detection and classification tasks are performed strictly using Classical Computer Vision techniques**, completely avoiding Deep Learning models.

## 👥 Game Structure

* **Participants:** The system is designed to manage a full game for **one physical Dealer and 1 to 2 active Players** simultaneously.
* The system provides independent tracking for each player, including: dedicated card detection, hand value calculation (distinguishing between soft and hard hands), physical bet detection, separate decision analysis, and individual bankroll management across multiple rounds.

## 📂 Project Structure
To ensure the system functions correctly, maintain the following directory structure:

```text
JACK_in_BLACK/
├── main.py              # Main entry point
├── camera_thread.py     # Multithreading & Camera logic
├── vision_engine.py     # OpenCV pipeline & Calibration
├── game_manager.py      # State machine logic
├── game_logic.py        # Blackjack rules & EV math
├── strategy_engine.py   # Strategy algorithms
├── ui_manager.py        # PyQt5 Dashboard
├── cards_classifier.py  # Image processing for cards
├── chips_classifier.py  # Image processing for chips
├── decision_chip_classifier.py # Decision tokens detection
├── requirements.txt     # Dependencies
├── .gitignore           # Git ignore rules
├── templates/           # Card templates
│   ├── ranks/           # Rank images
│   └── suits/           # Suit images
├── sounds/              # Audio feedback files
└── pictures/            # Logo
```
---

## 🌟 Key Technological Features

### 1. Hardware-Agnostic Vision & Environmental Adaptation
The system is designed as a true "Plug & Play" Production environment, capable of running on various hardware setups and under fluctuating casino conditions:
* **Dynamic Resolution Scaling:** Upon connection, the system probes the camera hardware to verify a strict 16:9 aspect ratio and dynamically calculates a global `SYSTEM_SCALE` factor. All algorithmic spatial parameters (distances, minimum areas, text sizing, and Hough Circle radii) automatically scale relative to the native resolution (from 720p up to 4K UHD).
* **Live Color Normalization:** To combat dynamic lighting and shadows, the system allows performing an initial felt calibration. It calculates an RGB correction vector (Gains) based on the specific green felt's median color, dynamically normalizing the entire video feed to ensure robust chip color classification regardless of room lighting.

### 2. Classical Computer Vision & Object Tracking
All detection pipelines were built from scratch using classic OpenCV algorithms, ensuring deterministic, low-latency execution without any reliance on Deep Learning models:
* **Robust Card Detection:** The pipeline uses contour detection and Homography (perspective warp transformations) to flatten physical cards into a normalized 2D plane. It then applies dynamic X/Y-axis histogram projection profiles to tightly crop the symbols, feeding them into an advanced Template Matching algorithm for high-accuracy rank and suit classification.
* **Stable Chip Recognition:** Utilizes HSV color space conversions, morphological operations, and the Hough Circle Transform to continuously locate and classify multi-colored chips. 
* **Noise Filtering & Memory Tracking:** To combat physical disturbances (moving hands, occlusion), we engineered a custom `ChipMemoryTracker`. It utilizes a voting mechanism combined with an **Exponential Moving Average (EMA)** algorithm to stabilize object coordinates and radii over time, completely eliminating UI jitter and ghost detections.
* **Spatial Centroid Tracking:** Prevents duplicate detections by actively measuring the Euclidean distance between newly detected cards and existing locked cards, ensuring flawless physical card counting.

### 3. Multi-Mode Strategy Engine & Mathematical Modeling
The decision-making system is strictly computational. It supports 3 game modes, ranging from basic gameplay to deep statistical analysis:
* **Regular Mode:** Free interactive gameplay without any algorithmic assistance, allowing players to practice independently.
* **The Perfect Gambler:** Provides real-time action recommendations (Hit, Stand, Double, Surrender) derived from deterministic Basic Strategy tables, comparing the player's hand against the dealer's upcard.
* **Super Computer Mode (Statistical & Computational Core):** The most advanced and mathematically rigorous mode of the project. Instead of relying on static heuristic tables, this engine computes the exact game statistics in real-time:
  * **Live Card Counting:** Maintains a mathematically exact inventory of the shoe across multiple rounds.
  * **Recursive Expected Value (EV) Maximization:** The algorithm dynamically computes the absolute Expected Value for every possible action. It constructs a dynamic state-space tree and performs a recursive depth-first simulation of all future outcomes. By multiplying the exact probability of drawing each remaining card by the maximized EV of the subsequent state, the engine mathematically maps the true probabilities of a Win, Loss, or Tie for any given decision.
  * **Pre-Round Risk Assessment:** Prior to dealing, the system calculates the statistical house/player edge using the "Effect of Removal" (EOR) principle. It then applies **Kelly Criterion** equations to recommend dynamically optimized bet sizing at the start of each round, maximizing long-term bankroll growth while minimizing the risk of ruin.

### 4. Zero-Latency Event-Driven Architecture
To ensure smooth real-time performance while executing heavy mathematical and visual computations, the system employs a robust, multithreaded design:
* **Strict 9-State Machine:** The entire game flow is governed by a centralized `GameManager` class operating as a strict 9-phase finite state machine (Calibration -> Betting -> Dealing -> Player Turns -> Comparison, etc.). This ensures flawless transitions and absolute protection against race conditions.
* **Producer-Consumer Camera Thread:** A dedicated micro-thread strictly pulls frames from the camera, dropping duplicates and managing buffers to prevent UI bottlenecks or stuttering.
* **Event Bus Synchronization:** Complete thread isolation between camera frame grabbing, heavy OpenCV image processing, and UI rendering. An internal **PyQt5 Event Bus** asynchronously and safely transmits data between the background threads and the Main UI Thread.

### 5. Advanced Real-Time UI/UX
A professional, immersive Dashboard built entirely with `PyQt5`:
* **Zero-Latency HUD Overlay:** The live video stream is augmented with a graphics layer that draws bounding boxes around cards, labels chip values, and projects the system's calculated conclusions directly onto the physical table feed.
* **Custom Data Visualizations:** Includes an interactive GUI Speedometer mapping real-time EV fluctuations and Donut Charts displaying the exact statistically calculated probability distributions for each action.
* **Micro-Interactions & Audio Feedback:** Features high-quality, event-driven audio cues (chip clinking, dealing sounds, winning chimes) via `pygame.mixer` and smooth rolling-number animations for bankroll updates, creating a highly polished, responsive casino atmosphere.

### 6. Dynamic Shoe & Realistic Cut Card Simulation
To authentically replicate a physical casino environment, the system supports tracking for varying shoe sizes (ranging from 1 to 8 decks) and implements a dynamic reshuffle mechanic based on statistical modeling:
* **Gaussian Cut Card Model:** Instead of triggering a reshuffle at a static deck penetration, the engine mathematically models the physical insertion of the "Cut Card" by a human dealer. It targets an industry-standard 75% shoe penetration (the mean) but applies a **Normal Distribution (`random.gauss`)** with a standard deviation of 10 cards to simulate natural human variance and inaccuracy.
* **Mathematical Clamping:** To ensure absolute game stability, the randomized threshold is strictly clamped. This prevents edge cases such as the dealer shuffling too early (enforcing a minimum 50% penetration) or running out of cards mid-hand, guaranteeing a resilient simulation.
* **Real-Time Inventory Tracking:** As cards are visually detected and dealt, they are subtracted from the system's global inventory in real-time. Once the exact running card count reaches the dynamically generated Cut Card threshold, the `GameManager` safely concludes the current round and forces a "Reshuffle" state.

### 7. Temporal Stabilization via Majority Vote
Processing live video introduces challenges not present in static image processing, such as temporary camera occlusions (e.g., a player's hand blocking the frame while placing a bet). To combat this, the vision engine implements a temporal stabilization buffer. It collects a sequence of consecutive frames and calculates a statistical **Majority Vote** (`Counter.most_common`) before locking in a detection. This ensures that transient visual interruptions are completely ignored by the logic engine.

### 8. System Health Telemetry & Resource Management
To ensure the system remains performant during prolonged sessions without overloading the CPU:
* A background telemetry mechanism continuously monitors pipeline health, tracking metrics such as dropped frames, successfully processed logic frames, and UI render rates. 
* The system actively limits the processing rate (FPS capping) and safely discards redundant frames using buffer locks, preventing thermal throttling and ensuring synchronization between the physical table and the UI.
---

## 🛠️ Physical Setup & Requirements

To run the system in an optimal physical environment, you will need:

* **Playing Surface:** A green felt mat to provide sufficient contrast for the color normalization algorithms.
* **Game Set:** Standard poker cards and colored casino chips matching the configured values (1, 5, 10, 25, 50).
* **Decision Chips:** 4 distinct tokens (e.g., ~3cm EVA foam circles) in specific colors:
* **Blue** for `HIT`
* **Pink** for `STAND`
* **Orange** for `DOUBLE DOWN`
* **Brown** for `SURRENDER`


* **Camera:** A smartphone running the `IP Webcam` app (or a similar IP camera).
* **Camera Placement:** The smartphone must be mounted **parallel (top-down) to the playing table at a height of approximately 70 cm** to minimize extreme perspective distortions and yield optimal detections.

---

## 💻 Tech Stack

* **Python 3.x**
* **OpenCV** - Classical image processing pipelines.
* **PyQt5** - Graphical User Interface and Multithreading management.
* **NumPy** - Fast matrix operations and mathematical computations.
* **Pygame** - Real-time audio cue management.

---

## 🚀 Installation & Usage

1. Clone the repository to your local machine:
```bash
git clone https://github.com/YourUsername/JACK_in_BLACK.git
cd JACK_in_BLACK

```


2. Install the required dependencies:
```bash
pip install -r requirements.txt

```


3. Ensure your smartphone and computer are connected to the same Wi-Fi network **(A wired USB connection is also supported and recommended for lower latency)** and launch the `IP Webcam` app.
4. Run the main application:
```bash
python main.py

```


5. Enter the IP address shown on your phone into the pop-up window, and follow the on-screen instructions to calibrate the Regions of Interest (ROIs) using your mouse.

6. **Gameplay Flow:**
Once the ROIs are calibrated, the game operates in a strict, continuous loop managed by the `GameManager`:
* **Initialization:** A popup dialog will appear. Enter the player names, starting bankrolls, deck count, and select your desired Game Mode.
* **Betting Phase:** A visual timer will start on the screen. Place your physical casino chips inside your designated betting ROI. The system will read and lock your bet once the timer expires.
* **Dealing Phase:** The physical dealer distributes the initial cards. The system will pause and wait until all required cards (2 per active player, 1 for the dealer) are clearly detected on the table.
* **Player Turns:** For each player, a decision timer will begin. Place one of the colored decision tokens (Blue=Hit, Pink=Stand, Orange=Double, Brown=Surrender) in your designated Decision ROI. The system will execute the action, prompt the dealer to draw a card if necessary, and calculate new probabilities.
* **Dealer Turn & Payouts:** The dealer reveals their hidden card and plays according to standard casino rules (Stands on Soft 17). The system evaluates all hands, updates the digital bankrolls, and flashes the round results.
* **Next Round:** Clear all cards from the table and click the "New Round" button on the UI dashboard to begin again.
---

## 👥 Project Team

* **Elad Cohen** - [LinkedIn Profile](https://www.linkedin.com/in/elad-cohen-758878304/)
* **Shira Atir** - [LinkedIn Profile](https://www.linkedin.com/in/shira-atir/)
* **Roni Dvir** - [LinkedIn Profile](https://www.linkedin.com/in/roni-dvir/)
* **Amir Zabari** - [LinkedIn Profile](https://www.linkedin.com/in/amir-tzabary/)