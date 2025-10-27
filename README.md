<<<<<<< HEAD
# RituCare
=======
# RituCare - Women's Health Companion

A comprehensive, privacy-first women's health tracking web application designed for GitHub Pages deployment. Track periods, assess PCOS risk, get nutrition guidance, and manage your health data all in one place.

## 🌟 Features

### **Period Tracking**
- Log period start/end dates with flow intensity
- Track symptoms and patterns
- View cycle statistics and trends
- Calendar visualization
- Cycle prediction

### **PCOS Assessment**
- Comprehensive risk assessment questionnaire
- Personalized recommendations
- Risk level tracking over time

### **Nutrition Guidance**
- Cycle-phase specific nutrition tips
- Meal suggestions based on cycle phase
- Hydration tracking

### **Profile Management**
- Personal health information
- Cycle preferences
- Data export/import
- Privacy-focused data management

## 🚀 Live Demo

Access the live application at: [https://yourusername.github.io/ritucare](https://yourusername.github.io/ritucare)

## 📱 Single-Page Application

This app is designed as a single-page application with tabbed navigation:
- **Dashboard**: Overview of your cycle, quick actions, and health insights
- **Period Tracker**: Detailed period logging and cycle analysis
- **PCOS Assessment**: Risk assessment and recommendations
- **Nutrition**: Cycle-synced nutrition tips and meal suggestions
- **Profile**: Personal settings and data management

## 🔒 Privacy & Data

- **Local Storage Only**: All data stays in your browser's localStorage
- **No Backend Required**: Perfect for GitHub Pages static hosting
- **Export/Import**: Backup and restore your data anytime
- **Data Control**: You own your health data

## 🛠️ Technical Stack

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **UI Framework**: Bootstrap 5
- **Icons**: Feather Icons
- **Charts**: Chart.js
- **Storage**: Browser localStorage
- **Hosting**: GitHub Pages

## 📁 Project Structure

```
ritucare/
├── index.html          # Main single-page application
├── css/
│   ├── style.css       # Main styles
│   └── components.css  # Component-specific styles
├── js/
│   ├── app.js          # Main application logic
│   ├── storage.js      # Data persistence
│   ├── cycle.js        # Period tracking logic
│   ├── pcos.js         # PCOS assessment logic
│   ├── nutrition.js    # Nutrition guidance
│   ├── charts.js       # Chart visualizations
│   └── ui.js           # UI utilities
├── data/
│   ├── nutrition-db.json   # Nutrition database
│   └── sample-data.json    # Sample data for testing
├── pages/              # (Legacy multi-page files - can be removed)
├── README.md           # This file
├── TODO.md             # Development notes
└── server.py           # (Development server - not needed for GitHub Pages)
```

## 🚀 Deployment to GitHub Pages

### Step 1: Create GitHub Repository
1. Create a new repository on GitHub
2. Name it `ritucare` or any name you prefer
3. Make it public for free hosting

### Step 2: Upload Files
```bash
# Clone your repository
git clone https://github.com/yourusername/ritucare.git
cd ritucare

# Copy all files from this project
# (copy index.html, css/, js/, data/ folders)

# Commit and push
git add .
git commit -m "Initial commit - RituCare Women's Health App"
git push origin main
```

### Step 3: Enable GitHub Pages
1. Go to your repository on GitHub
2. Click **Settings** tab
3. Scroll down to **Pages** section
4. Under **Source**, select **Deploy from a branch**
5. Under **Branch**, select **main** (or your default branch)
6. Click **Save**

### Step 4: Access Your App
Your app will be available at: `https://yourusername.github.io/ritucare`

## 🔧 Local Development

For local development and testing:

```bash
# Start a local server
python -m http.server 8000

# Or use any static file server
# Access at http://localhost:8000
```

## 📊 Data Persistence

All user data is stored locally in the browser:
- **Profile Information**: Name, age, height, weight
- **Period Logs**: Start/end dates, flow intensity, symptoms
- **PCOS Assessments**: Risk scores and recommendations
- **Nutrition Logs**: Meal tracking and preferences
- **Settings**: App preferences and cycle settings

### Exporting Data
1. Go to Profile tab
2. Click "Export Data"
3. Save the JSON file to backup your data

### Importing Data
1. Go to Profile tab
2. Click "Import Data"
3. Select your previously exported JSON file

## 🎨 Customization

### Colors & Theme
Edit `css/style.css` to customize colors, fonts, and styling.

### Features
Modify `js/app.js` to add new features or change existing functionality.

### Nutrition Database
Update `data/nutrition-db.json` to add more nutrition tips and meal suggestions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Built with Bootstrap 5 for responsive design
- Chart.js for data visualization
- Feather Icons for beautiful iconography
- Inspired by the need for privacy-focused health tracking

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the browser console for errors
- Ensure JavaScript is enabled in your browser

---

**Remember**: This app is for informational purposes only and should not replace professional medical advice. Consult with healthcare providers for medical concerns.
>>>>>>> master
