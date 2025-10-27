# RituCare - Women's Health Companion

A comprehensive, local-first web application for women's health tracking, built with modern web technologies.

## Features

### 🗓️ Period Tracker
- Log periods with start/end dates, flow intensity, and symptoms
- Visual calendar with period predictions and fertile windows
- Cycle statistics and trend analysis
- Symptom frequency tracking

### 🩺 PCOS Assessment
- Comprehensive risk assessment questionnaire
- BMI calculation and symptom evaluation
- Risk level visualization with gauge charts
- Assessment history tracking

### 🥗 Nutrition Guide
- Phase-based nutrition recommendations
- Daily meal suggestions synced to menstrual cycle
- Shopping list generation
- Personalized tips for each cycle phase

### 📊 Dashboard
- Today's cycle information and phase
- Quick actions for logging
- PCOS risk overview
- Mini calendar view
- Hydration tracking

### 👤 Profile Management
- Personal information storage
- Cycle preferences customization
- Data export/import functionality
- Privacy-focused (all data stays local)

## Technology Stack

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **UI Framework**: Bootstrap 5
- **Charts**: Chart.js
- **Icons**: Feather Icons
- **Storage**: Browser localStorage (offline-first)
- **Fonts**: Google Fonts (Inter)

## Getting Started

### Prerequisites
- Modern web browser with JavaScript enabled
- No server required (runs entirely in browser)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/10anshika/RituCare.git
   cd RituCare
   ```

2. **Start local server:**
   ```bash
   # Using Python (recommended)
   python -m http.server 8000

   # Or using Node.js
   npx http-server -p 8000

   # Or using PHP
   php -S localhost:8000
   ```

3. **Open in browser:**
   Navigate to `http://localhost:8000` or `http://127.0.0.1:8000`

### Usage

1. **First Time Setup:**
   - Go to Profile tab and enter your basic information
   - Set your cycle preferences

2. **Start Tracking:**
   - Use "Log New Period" to record your periods
   - Complete the PCOS assessment for risk evaluation
   - View nutrition recommendations based on your cycle

3. **Data Management:**
   - All data is stored locally in your browser
   - Export data for backup using Profile → Data Management
   - Import data to restore from backup

## Architecture

### File Structure
```
RituCare/
├── index.html          # Main application (single-page)
├── pages/              # Additional page templates
│   ├── tracker.html
│   ├── pcos.html
│   ├── nutrition.html
│   └── profile.html
├── js/                 # JavaScript modules
│   ├── app.js          # Main application logic
│   ├── storage.js      # localStorage management
│   ├── cycle.js        # Cycle calculations & predictions
│   ├── pcos.js         # PCOS assessment logic
│   ├── nutrition.js    # Nutrition recommendations
│   ├── charts.js       # Chart.js wrappers
│   └── ui.js           # UI components & utilities
├── css/                # Stylesheets
│   ├── style.css       # Theme & base styles
│   └── components.css  # Component-specific styles
├── data/               # Static data files
│   ├── nutrition-db.json
│   └── sample-data.json
└── assets/             # Static assets (icons, images)
```

### Key Components

- **StorageManager**: Handles all localStorage operations
- **CycleManager**: Period tracking, predictions, and cycle analysis
- **PCOSManager**: Risk assessment calculations and scoring
- **NutritionManager**: Phase-based nutrition recommendations
- **ChartsManager**: Chart.js integration for data visualization
- **RituCareApp**: Main application controller

## Privacy & Security

- **Local-First**: All data stored in browser localStorage
- **No External APIs**: No data sent to servers (unless optional features enabled)
- **Privacy-Focused**: No tracking, analytics, or data collection
- **Offline Capable**: Works without internet connection

## Browser Support

- Chrome 80+
- Firefox 75+
- Safari 13+
- Edge 80+

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with Bootstrap, Chart.js, and Feather Icons
- Inspired by the need for privacy-focused health tracking
- Designed for accessibility and ease of use

## Support

For questions or issues:
- Open an issue on GitHub
- Check the documentation in this README
- Ensure you're using a modern web browser

---

**Note**: This application is for informational purposes only and should not replace professional medical advice. Always consult with healthcare providers for medical concerns.
