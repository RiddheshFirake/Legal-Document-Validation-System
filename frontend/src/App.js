import React, { useState, useEffect } from 'react';
import {
  Container,
  AppBar,
  Toolbar,
  Typography,
  Tabs,
  Tab,
  Box,
  Alert,
  Chip,
  Grid,
} from '@mui/material';
import {
  Description,
  DynamicFeed,
  Timeline,
  Settings,
  CheckCircle,
  Error,
} from '@mui/icons-material';

import DocumentUpload from './components/DocumentUpload';
import ValidationResults from './components/ValidationResults';
import BatchUpload from './components/BatchUpload';
import JobStatus from './components/JobStatus';
import SystemStatus from './components/SystemStatus';
import { api } from './services/api';
import './styles/App.css';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';

// Enhanced dark theme for perfect glassmorphism
// Update the darkTheme in App.js
const darkTheme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00B0F4', // Endpointify blue
    },
    secondary: {
      main: '#00FFE5', // Endpointify accent
    },
    background: {
      default: '#000000', // Pure black
      paper: 'rgba(0, 176, 244, 0.05)',
    },
    text: {
      primary: '#ffffff',
      secondary: 'rgba(255, 255, 255, 0.7)',
    },
  },
  components: {
    MuiPaper: {
      styleOverrides: {
        root: {
          background: 'rgba(0, 176, 244, 0.05)',
          backdropFilter: 'blur(16px)',
          border: '1px solid rgba(0, 176, 244, 0.2)',
        },
      },
    },
  },
});


function TabPanel({ children, value, index, ...other }) {
  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`tabpanel-${index}`}
      aria-labelledby={`tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3, backgroundColor: 'transparent' }}>
          {children}
        </Box>
      )}
    </div>
  );
}

function App() {
  const [tabValue, setTabValue] = useState(0);
  const [systemHealth, setSystemHealth] = useState(null);
  const [validationResult, setValidationResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    checkSystemHealth();
  }, []);

  const checkSystemHealth = async () => {
    try {
      const response = await api.healthCheck();
      setSystemHealth(response.data);
    } catch (error) {
      setError('Failed to connect to the API server. Please ensure the backend is running.');
    }
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
    setError(null);
  };

  const handleValidationComplete = (result) => {
    setValidationResult(result);
  };

  return (
    <ThemeProvider theme={darkTheme}>
      <CssBaseline />
      <div className="App">
        {/* Enhanced AppBar with glassmorphism */}
        <AppBar 
          position="static" 
          sx={{
            background: 'rgba(255, 255, 255, 0.08)',
            backdropFilter: 'blur(20px)',
            borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
            boxShadow: '0 4px 30px rgba(0, 0, 0, 0.1)',
          }}
        >
          <Toolbar>
            <Description sx={{ mr: 2, filter: 'drop-shadow(0 2px 4px rgba(0, 229, 255, 0.5))' }} />
            <Typography 
              variant="h6" 
              component="div" 
              sx={{ 
                flexGrow: 1,
                fontWeight: 600,
                textShadow: '0 2px 10px rgba(0, 0, 0, 0.3)',
                background: 'linear-gradient(45deg, #ffffff, #00e5ff)',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                color: 'transparent',
              }}
            >
              Legal Document Validation System
            </Typography>
            {systemHealth && (
              <Chip
                icon={systemHealth.status === 'healthy' ? <CheckCircle /> : <Error />}
                label={systemHealth.status === 'healthy' ? 'System Healthy' : 'System Error'}
                color={systemHealth.status === 'healthy' ? 'success' : 'error'}
                variant="outlined"
                sx={{ 
                  color: 'white', 
                  borderColor: 'rgba(255, 255, 255, 0.3)',
                  background: 'rgba(255, 255, 255, 0.1)',
                  backdropFilter: 'blur(10px)',
                  '&:hover': {
                    background: 'rgba(255, 255, 255, 0.15)',
                  }
                }}
              />
            )}
          </Toolbar>
        </AppBar>

        <Container maxWidth="xl" sx={{ mt: 3, mb: 3 }}>
          {/* Enhanced Error Alert */}
          {error && (
            <Alert 
              severity="error" 
              sx={{ 
                mb: 3,
                background: 'rgba(244, 67, 54, 0.15)',
                backdropFilter: 'blur(15px)',
                border: '1px solid rgba(244, 67, 54, 0.3)',
                borderRadius: '16px',
                color: '#ffffff',
                '& .MuiAlert-icon': {
                  color: '#ff6b6b',
                }
              }} 
              onClose={() => setError(null)}
            >
              {error}
            </Alert>
          )}

          {/* Main Glass Container */}
          <div 
            className="glass-card" 
            style={{ 
              padding: 0, 
              overflow: 'hidden',
              background: 'rgba(255, 255, 255, 0.08)',
              backdropFilter: 'blur(16px)',
              borderRadius: '24px',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              boxShadow: '0 8px 32px 0 rgba(31, 38, 135, 0.37)',
            }}
          >
            {/* Enhanced Tabs with Glass Effect */}
            <Tabs
              value={tabValue}
              onChange={handleTabChange}
              variant="fullWidth"
              indicatorColor="primary"
              textColor="primary"
              sx={{
                background: 'rgba(255, 255, 255, 0.05)',
                borderRadius: '20px 20px 0 0',
                '& .MuiTab-root': {
                  color: 'rgba(255, 255, 255, 0.7)',
                  fontWeight: 500,
                  textTransform: 'none',
                  minHeight: '64px',
                  transition: 'all 0.3s ease',
                  '&.Mui-selected': {
                    color: '#00e5ff',
                    fontWeight: 600,
                    background: 'rgba(0, 229, 255, 0.1)',
                    borderRadius: '12px',
                    margin: '8px',
                  },
                  '&:hover': {
                    background: 'rgba(255, 255, 255, 0.05)',
                    borderRadius: '12px',
                    margin: '8px',
                  }
                },
                '& .MuiTabs-indicator': {
                  background: 'linear-gradient(90deg, #00e5ff, #18ffff)',
                  height: '3px',
                  borderRadius: '2px',
                }
              }}
            >
              <Tab icon={<Description />} label="Single Document" />
              <Tab icon={<DynamicFeed />} label="Batch Processing" />
              <Tab icon={<Timeline />} label="Job Status" />
              <Tab icon={<Settings />} label="System Status" />
            </Tabs>

            {/* Tab Content with Glass Panels */}
            <TabPanel value={tabValue} index={0}>
              <Grid container spacing={3}>
                <Grid item xs={12} md={6}>
                  <div 
                    className="glass-card" 
                    style={{ 
                      padding: '24px',
                      background: 'rgba(255, 255, 255, 0.05)',
                      backdropFilter: 'blur(10px)',
                      borderRadius: '20px',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                    }}
                  >
                    <DocumentUpload
                      onValidationComplete={handleValidationComplete}
                      onError={setError}
                      loading={loading}
                      setLoading={setLoading}
                    />
                  </div>
                </Grid>
                <Grid item xs={12} md={6}>
                  <div 
                    className="glass-card" 
                    style={{ 
                      padding: '24px',
                      background: 'rgba(255, 255, 255, 0.05)',
                      backdropFilter: 'blur(10px)',
                      borderRadius: '20px',
                      border: '1px solid rgba(255, 255, 255, 0.1)',
                    }}
                  >
                    <ValidationResults
                      result={validationResult}
                      loading={loading}
                    />
                  </div>
                </Grid>
              </Grid>
            </TabPanel>

            <TabPanel value={tabValue} index={1}>
              <div 
                className="glass-card" 
                style={{ 
                  padding: '24px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  backdropFilter: 'blur(10px)',
                  borderRadius: '20px',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                }}
              >
                <BatchUpload onError={setError} />
              </div>
            </TabPanel>

            <TabPanel value={tabValue} index={2}>
              <div 
                className="glass-card" 
                style={{ 
                  padding: '24px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  backdropFilter: 'blur(10px)',
                  borderRadius: '20px',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                }}
              >
                <JobStatus onError={setError} />
              </div>
            </TabPanel>

            <TabPanel value={tabValue} index={3}>
              <div 
                className="glass-card" 
                style={{ 
                  padding: '24px',
                  background: 'rgba(255, 255, 255, 0.05)',
                  backdropFilter: 'blur(10px)',
                  borderRadius: '20px',
                  border: '1px solid rgba(255, 255, 255, 0.1)',
                }}
              >
                <SystemStatus
                  systemHealth={systemHealth}
                  onRefresh={checkSystemHealth}
                  onError={setError}
                />
              </div>
            </TabPanel>
          </div>
        </Container>
      </div>
    </ThemeProvider>
  );
}

export default App;
