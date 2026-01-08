# 🏆 Ultimate Chess AI Project

## 🚀 Quick Start (One Command!)

```bash
python ultimate_launcher.py
```

This single command gives you access to everything:
- 🧠 **Intelligent AI Training** - Automatically processes your Lichess database and trains the neural network
- 🎮 **Chess GUI** - Desktop chess game with parallel CPU/GPU evaluation
- 🌐 **Web Interface** - Browser-based chess platform with user accounts
- 📊 **Training Analytics** - Monitor AI learning progress
- 💾 **Auto-Save/Resume** - Never lose training progress

## 🗃️ Database Training

1. **Add your PGN database** (like `lichess_db_standard_rated_2017-10.pgn`)
2. **Run the launcher**: `python ultimate_launcher.py`
3. **Select option 1**: Train AI Model
4. **The system automatically**:
   - Processes millions of chess games
   - Trains the neural network with checkpoints
   - Runs self-play improvement
   - Saves the final trained model
   - Resumes if interrupted

## 🎯 Features

### Smart Training Pipeline
- **Database Processing**: Efficiently handles large PGN files (GB+ sizes)
- **Neural Network**: AlphaZero-style CNN with value and policy heads
- **Self-Play Training**: AI improves by playing against itself
- **Checkpointing**: Save/resume training anytime
- **Progress Monitoring**: Real-time training analytics

### Game Interfaces
- **Desktop GUI**: tkinter-based chess game with AI opponent
- **Web Platform**: Flask-based online chess with user accounts
- **Multiple AI Levels**: From simple evaluation to trained neural network

### Technical Features
- **Parallel Evaluation**: CPU and GPU benchmarking
- **Move Validation**: Full chess rules implementation
- **Sound Effects**: Audio feedback for moves and captures
- **Game Logging**: Detailed game history and analysis

## 📋 Libraries Used

Essential libraries in your code:

**GUI & Interface:**
- `tkinter` - Desktop chess interface
- `flask` / `flask-socketio` - Web interface and real-time updates

**Chess Logic:**
- `chess` (python-chess) - Complete chess rules and board representation

**AI & Machine Learning:**
- `torch` (PyTorch) - Neural network training and inference
- `numpy` - Mathematical operations and array processing

**Data Processing:**
- `threading` - Parallel CPU/GPU evaluation
- `pickle` - Saving/loading training data
- `sqlite3` - Game history and user data storage

**Audio & Multimedia:**
- `pygame` - Sound effects for chess moves

**Utilities:**
- `logging` - Training progress and debugging
- `datetime` - Timestamps and game timing
- `platform` - System information for benchmarks

## 🛠️ Installation

```bash
# Essential packages for offline training
pip install torch numpy python-chess pygame

# Additional packages for web features (optional)
pip install flask flask-socketio matplotlib pandas seaborn

# Run the project
python ultimate_launcher.py
```

## 🔄 **Training Modes**

### 🖥️ **Offline Training (No Network Required)**
✅ **Local Database Training** - Uses your existing PGN files  
✅ **Self-Play Training** - AI improves by playing against itself  
✅ **Model Checkpointing** - All progress saved locally  
✅ **Desktop Chess GUI** - Play against trained AI offline  

**Requirements:** Only `torch`, `numpy`, `python-chess`, `pygame`

### 🌐 **Online Features (Network Optional)**
🌍 **Web Interface** - Browser-based chess platform  
📊 **Training Analytics** - Advanced progress visualization  
🗃️ **Database Downloads** - Get new PGN files from Lichess  

**Additional Requirements:** `flask`, `flask-socketio`, `matplotlib`, `pandas`, `seaborn`

## 🎮 How to Use

### 1. Training Your AI (First Time)
1. Place your Lichess PGN database in the project folder
2. Run: `python ultimate_launcher.py`
3. Choose option 1: "Train AI Model"
4. Select your database file
5. Wait for training to complete (can take hours/days)
6. Model is automatically saved for future use

### 2. Playing Chess
**Desktop Version:**
- Choose option 2 in launcher
- Select game mode (manual/vs_ai/engine)
- Choose your color and difficulty
- Play using drag-and-drop

**Web Version:**
- Choose option 3 in launcher
- Open browser to localhost:5000
- Create account or login
- Play online with full features

### 3. Monitoring Training
- Choose option 4 for training analytics
- View loss curves and model performance
- Check training logs in `training_logs/` folder

## 📁 Project Structure

**Core Files:**
- `ultimate_launcher.py` - 🚀 **Main launcher (start here)**
- `neural_network.py` - AI neural network architecture
- `chess_game.py` - Desktop chess GUI
- `enhanced_web_interface.py` - Web chess platform

**Training System:**
- `intelligent_training_pipeline.py` - Smart training orchestrator
- `smart_database_parser.py` - Efficient PGN database processor
- `self_play_training.py` - Self-play improvement system
- `mcts.py` - Monte Carlo Tree Search algorithm

**Supporting:**
- `training_monitor.py` - Progress tracking and analytics
- `data_integration.py` - External data loading utilities
- `model_evaluation.py` - AI performance testing
   
   pip install python-chess torch pygame







Attributes used for AI judgment (evaluation model):
The AI will judge a chess position based on features (attributes) like:

♟️ Material balance – Total value of pieces on each side (e.g., pawn = 1, queen = 9).

🏰 King safety – Whether the king is well protected or exposed.

🔄 Mobility – Number of legal moves available (activity of pieces).

💣 Threats – Are pieces attacking valuable opponent pieces?

🧱 Pawn structure – Doubled, isolated, or passed pawns.

📏 Piece positioning – How well centralized or active the pieces are.

🔗 Control of center – Who controls important squares like e4, d4, e5, d5.

🕳️ Weak squares – Are there holes in the opponent’s position?

♜ Rook activity – Are rooks on open or semi-open files?

⚖️ Game phase – Opening, middlegame, or endgame (affects how above features are weighed).
   
