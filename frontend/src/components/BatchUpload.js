import React, { useState, useCallback } from 'react';
import {
  Typography,
  Button,
  Box,
  LinearProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
  FormControlLabel,
  Switch,
  Alert,
  Grid,
  Slider,
} from '@mui/material';
import {
  CloudUpload,
  Delete,
  CheckCircle,
  Error,
  Warning,
  Visibility,
  DynamicFeed,
  Assessment,
} from '@mui/icons-material';
import { api } from '../services/api';

const BatchUpload = ({ onError }) => {
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [options, setOptions] = useState({
    returnDetailed: false,
    maxConcurrent: 3,
  });

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    const files = Array.from(e.dataTransfer.files);
    setSelectedFiles(prev => [...prev, ...files]);
  }, []);

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleFileChange = (e) => {
    if (e.target.files) {
      const files = Array.from(e.target.files);
      setSelectedFiles(prev => [...prev, ...files]);
    }
  };

  const removeFile = (index) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index));
  };

  const clearAllFiles = () => {
    setSelectedFiles([]);
  };

  const validateBatch = async () => {
    if (selectedFiles.length === 0) {
      onError('Please select at least one file');
      return;
    }

    setLoading(true);
    setResults(null);
    
    try {
      const response = await api.validateBatch(selectedFiles, options);
      setResults(response.data);
      onError(null);
    } catch (error) {
      onError(error.response?.data?.detail || 'Batch validation failed');
    } finally {
      setLoading(false);
    }
  };

  const getResultIcon = (result) => {
    if (!result.success) return <Error sx={{ color: '#ef5350', filter: 'drop-shadow(0 2px 4px rgba(239, 83, 80, 0.5))' }} />;
    if (result.decision?.is_legal) return <CheckCircle sx={{ color: '#66bb6a', filter: 'drop-shadow(0 2px 4px rgba(102, 187, 106, 0.5))' }} />;
    return <Warning sx={{ color: '#ffab40', filter: 'drop-shadow(0 2px 4px rgba(255, 171, 64, 0.5))' }} />;
  };

  const getResultChip = (result) => {
    if (!result.success) {
      return (
        <Chip 
          label="Error" 
          size="small" 
          sx={{
            background: 'rgba(244, 67, 54, 0.2)',
            color: '#ffffff',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(244, 67, 54, 0.4)',
            fontWeight: 600,
          }}
        />
      );
    }
    const isLegal = result.decision?.is_legal;
    
    return (
      <Chip
        label={isLegal ? 'LEGAL' : 'NOT LEGAL'}
        size="small"
        sx={{
          background: isLegal 
            ? 'rgba(76, 175, 80, 0.2)' 
            : 'rgba(244, 67, 54, 0.2)',
          color: '#ffffff',
          backdropFilter: 'blur(10px)',
          border: isLegal 
            ? '1px solid rgba(76, 175, 80, 0.4)' 
            : '1px solid rgba(244, 67, 54, 0.4)',
          fontWeight: 600,
        }}
      />
    );
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <Box sx={{ height: '100%' }}>
      {/* Glass Title */}
      <Typography 
        variant="h5" 
        gutterBottom 
        sx={{ 
          color: '#ffffff',
          fontWeight: 600,
          mb: 3,
          textAlign: 'center',
          textShadow: '0 2px 10px rgba(0, 0, 0, 0.3)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 1,
        }}
      >
        <DynamicFeed sx={{ color: '#00e5ff', filter: 'drop-shadow(0 2px 4px rgba(0, 229, 255, 0.5))' }} />
        Batch Document Processing
      </Typography>

      {/* Enhanced Upload Container */}
      <Box 
        sx={{ 
          background: 'rgba(255, 255, 255, 0.05)',
          borderRadius: '20px',
          backdropFilter: 'blur(15px)',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          p: 3, 
          mb: 3,
          boxShadow: '0 8px 32px rgba(31, 38, 135, 0.37)',
        }}
      >
        {/* Enhanced File Upload Area */}
        <Box
          className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
          sx={{
            border: dragActive 
              ? '2px dashed rgba(0, 229, 255, 0.8)' 
              : '2px dashed rgba(255, 255, 255, 0.3)',
            borderRadius: '20px',
            p: 4,
            textAlign: 'center',
            background: dragActive 
              ? 'rgba(0, 229, 255, 0.15)' 
              : 'rgba(255, 255, 255, 0.05)',
            backdropFilter: 'blur(15px)',
            cursor: 'pointer',
            transition: 'all 0.4s cubic-bezier(0.23, 1, 0.320, 1)',
            mb: 3,
            position: 'relative',
            overflow: 'hidden',
            '&::before': {
              content: '""',
              position: 'absolute',
              top: 0,
              left: '-100%',
              width: '100%',
              height: '100%',
              background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent)',
              transition: 'left 0.5s ease',
            },
            '&:hover': {
              background: 'rgba(255, 255, 255, 0.12)',
              borderColor: 'rgba(0, 229, 255, 0.5)',
              boxShadow: '0 0 30px rgba(0, 229, 255, 0.3)',
              transform: 'translateY(-2px)',
              '&::before': {
                left: '100%',
              }
            },
          }}
          onDrop={handleDrop}
          onDragOver={handleDrag}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onClick={() => document.getElementById('batch-file-input').click()}
        >
          <DynamicFeed 
            sx={{ 
              fontSize: 64, 
              color: dragActive ? '#00e5ff' : 'rgba(255, 255, 255, 0.7)', 
              mb: 2,
              filter: dragActive 
                ? 'drop-shadow(0 4px 12px rgba(0, 229, 255, 0.6))'
                : 'drop-shadow(0 4px 8px rgba(0, 229, 255, 0.3))',
              transition: 'all 0.3s ease',
            }} 
          />
          <Typography 
            variant="h6" 
            gutterBottom 
            sx={{ 
              color: '#ffffff', 
              fontWeight: 600,
              textShadow: '0 2px 4px rgba(0, 0, 0, 0.3)',
            }}
          >
            Drop Multiple Files or Click to Upload
          </Typography>
          <Typography 
            variant="body2" 
            sx={{ 
              color: 'rgba(255, 255, 255, 0.7)',
              textShadow: '0 1px 2px rgba(0, 0, 0, 0.2)',
            }}
          >
            Select up to 10 documents for batch processing
          </Typography>
          <input
            id="batch-file-input"
            type="file"
            accept=".pdf,.docx,.doc,.jpg,.jpeg,.png,.txt"
            onChange={handleFileChange}
            multiple
            style={{ display: 'none' }}
          />
        </Box>

        {/* Enhanced Options Section */}
        <Box 
          sx={{ 
            mb: 3,
            p: 2,
            background: 'rgba(255, 255, 255, 0.05)',
            borderRadius: '16px',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
          }}
        >
          <Typography 
            variant="subtitle2" 
            gutterBottom 
            sx={{ 
              color: 'rgba(255, 255, 255, 0.9)',
              fontWeight: 600,
              mb: 2,
            }}
          >
            Processing Options
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={options.returnDetailed}
                    onChange={(e) =>
                      setOptions({ ...options, returnDetailed: e.target.checked })
                    }
                    sx={{
                      '& .MuiSwitch-switchBase.Mui-checked': {
                        color: '#00e5ff',
                      },
                      '& .MuiSwitch-switchBase.Mui-checked + .MuiSwitch-track': {
                        backgroundColor: '#00e5ff',
                      },
                    }}
                  />
                }
                label={
                  <Typography sx={{ color: 'rgba(255, 255, 255, 0.9)', fontWeight: 500 }}>
                    Detailed Analysis
                  </Typography>
                }
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <Typography 
                variant="body2" 
                sx={{ 
                  color: 'rgba(255, 255, 255, 0.9)', 
                  fontWeight: 600,
                  mb: 1 
                }}
              >
                Max Concurrent: {options.maxConcurrent}
              </Typography>
              <Slider
                value={options.maxConcurrent}
                onChange={(e, value) =>
                  setOptions({ ...options, maxConcurrent: value })
                }
                min={1}
                max={5}
                step={1}
                marks
                valueLabelDisplay="auto"
                sx={{
                  color: '#00e5ff',
                  '& .MuiSlider-track': {
                    background: 'linear-gradient(90deg, #00e5ff, #18ffff)',
                  },
                  '& .MuiSlider-thumb': {
                    background: 'linear-gradient(135deg, #00e5ff, #18ffff)',
                    border: '2px solid rgba(255, 255, 255, 0.2)',
                    boxShadow: '0 4px 12px rgba(0, 229, 255, 0.4)',
                  },
                  '& .MuiSlider-mark': {
                    background: 'rgba(255, 255, 255, 0.3)',
                  },
                  '& .MuiSlider-markActive': {
                    background: '#00e5ff',
                  }
                }}
              />
            </Grid>
          </Grid>
        </Box>

        {/* Enhanced Selected Files Display */}
        {selectedFiles.length > 0 && (
          <Box 
            sx={{ 
              mb: 3,
              background: 'rgba(255, 255, 255, 0.05)',
              borderRadius: '16px',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              overflow: 'hidden',
            }}
          >
            <Box sx={{ p: 2, display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <Typography 
                variant="h6" 
                sx={{ 
                  color: '#ffffff',
                  fontWeight: 600,
                }}
              >
                Selected Files ({selectedFiles.length})
              </Typography>
              <Button
                size="small"
                onClick={clearAllFiles}
                startIcon={<Delete />}
                sx={{
                  color: 'rgba(255, 255, 255, 0.7)',
                  '&:hover': {
                    color: '#ff6b6b',
                    background: 'rgba(255, 107, 107, 0.1)',
                  }
                }}
              >
                Clear All
              </Button>
            </Box>
            <TableContainer>
              <Table size="small">
                <TableHead>
                  <TableRow sx={{ background: 'rgba(255, 255, 255, 0.1)' }}>
                    <TableCell sx={{ color: '#ffffff', fontWeight: 600 }}>File Name</TableCell>
                    <TableCell sx={{ color: '#ffffff', fontWeight: 600 }}>Size</TableCell>
                    <TableCell sx={{ color: '#ffffff', fontWeight: 600 }}>Action</TableCell>
                  </TableRow>
                </TableHead>
                <TableBody>
                  {selectedFiles.map((file, index) => (
                    <TableRow 
                      key={index}
                      sx={{
                        '&:hover': {
                          background: 'rgba(0, 229, 255, 0.1)',
                        }
                      }}
                    >
                      <TableCell sx={{ color: 'rgba(255, 255, 255, 0.9)' }}>
                        {file.name}
                      </TableCell>
                      <TableCell sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                        {formatFileSize(file.size)}
                      </TableCell>
                      <TableCell>
                        <Button
                          size="small"
                          onClick={() => removeFile(index)}
                          startIcon={<Delete />}
                          sx={{
                            color: 'rgba(255, 255, 255, 0.7)',
                            '&:hover': {
                              color: '#ff6b6b',
                              background: 'rgba(255, 107, 107, 0.1)',
                            }
                          }}
                        >
                          Remove
                        </Button>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            </TableContainer>
          </Box>
        )}

        {/* Enhanced Process Button */}
        <Button
          variant="contained"
          size="large"
          fullWidth
          onClick={validateBatch}
          disabled={loading || selectedFiles.length === 0}
          startIcon={<CloudUpload />}
          sx={{
            background: 'linear-gradient(135deg, rgba(0, 229, 255, 0.3), rgba(24, 255, 255, 0.3))',
            border: '1px solid rgba(0, 229, 255, 0.4)',
            color: '#ffffff',
            borderRadius: '16px',
            backdropFilter: 'blur(10px)',
            textTransform: 'none',
            fontWeight: 700,
            fontSize: '1.1rem',
            padding: '12px 24px',
            boxShadow: '0 4px 20px rgba(0, 229, 255, 0.3)',
            transition: 'all 0.3s ease',
            '&:hover:not(:disabled)': {
              background: 'linear-gradient(135deg, rgba(0, 229, 255, 0.5), rgba(24, 255, 255, 0.5))',
              borderColor: 'rgba(0, 229, 255, 0.6)',
              boxShadow: '0 8px 30px rgba(0, 229, 255, 0.5)',
              transform: 'translateY(-2px)',
            },
            '&:disabled': {
              background: 'rgba(255, 255, 255, 0.1)',
              color: 'rgba(255, 255, 255, 0.5)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
            }
          }}
        >
          {loading ? 'Processing Batch...' : selectedFiles.length > 0 ? `Process ${selectedFiles.length} Files` : 'Select Files to Process'}
        </Button>

        {/* Enhanced Loading Section */}
        {loading && (
          <Box sx={{ mt: 3 }}>
            <LinearProgress 
              sx={{
                background: 'rgba(255, 255, 255, 0.1)',
                borderRadius: '10px',
                backdropFilter: 'blur(10px)',
                height: '8px',
                '& .MuiLinearProgress-bar': {
                  background: 'linear-gradient(90deg, #00e5ff, #18ffff)',
                  borderRadius: '10px',
                }
              }}
            />
            <Typography 
              variant="body2" 
              align="center" 
              sx={{ 
                mt: 2,
                color: 'rgba(255, 255, 255, 0.8)',
                fontWeight: 500,
                textShadow: '0 1px 2px rgba(0, 0, 0, 0.2)',
              }}
            >
              🔄 Processing {selectedFiles.length} documents in parallel...
            </Typography>
          </Box>
        )}
      </Box>

      {/* Enhanced Results Section */}
      {results && (
        <Box 
          sx={{ 
            background: 'rgba(255, 255, 255, 0.05)',
            borderRadius: '20px',
            backdropFilter: 'blur(15px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            p: 3,
            boxShadow: '0 8px 32px rgba(31, 38, 135, 0.37)',
          }}
        >
          <Typography 
            variant="h5" 
            gutterBottom
            sx={{
              color: '#ffffff',
              fontWeight: 600,
              mb: 3,
              textAlign: 'center',
              textShadow: '0 2px 10px rgba(0, 0, 0, 0.3)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              gap: 1,
            }}
          >
            <Assessment sx={{ color: '#00e5ff', filter: 'drop-shadow(0 2px 4px rgba(0, 229, 255, 0.5))' }} />
            Batch Results
          </Typography>

          <Alert 
            severity="success" 
            sx={{ 
              mb: 3,
              background: 'rgba(76, 175, 80, 0.15)',
              backdropFilter: 'blur(15px)',
              border: '1px solid rgba(76, 175, 80, 0.3)',
              borderRadius: '16px',
              color: '#ffffff',
              '& .MuiAlert-icon': {
                color: '#81c784',
              }
            }}
          >
            ✅ Successfully processed {results.total_files} files with batch analysis
          </Alert>

          <TableContainer 
            sx={{
              background: 'rgba(255, 255, 255, 0.05)',
              borderRadius: '16px',
              backdropFilter: 'blur(10px)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              overflow: 'hidden',
            }}
          >
            <Table>
              <TableHead>
                <TableRow sx={{ background: 'rgba(255, 255, 255, 0.1)' }}>
                  <TableCell sx={{ color: '#ffffff', fontWeight: 600 }}>File</TableCell>
                  <TableCell sx={{ color: '#ffffff', fontWeight: 600 }}>Result</TableCell>
                  <TableCell sx={{ color: '#ffffff', fontWeight: 600 }}>Confidence</TableCell>
                  <TableCell sx={{ color: '#ffffff', fontWeight: 600 }}>Time</TableCell>
                  <TableCell sx={{ color: '#ffffff', fontWeight: 600 }}>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {results.results.map((result, index) => (
                  <TableRow 
                    key={index}
                    sx={{
                      '&:hover': {
                        background: 'rgba(0, 229, 255, 0.1)',
                      }
                    }}
                  >
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        {getResultIcon(result)}
                        <Box sx={{ ml: 1 }}>
                          <Typography 
                            variant="body2"
                            sx={{ 
                              color: 'rgba(255, 255, 255, 0.9)',
                              fontWeight: 500,
                            }}
                          >
                            {selectedFiles[index]?.name || `File ${index + 1}`}
                          </Typography>
                        </Box>
                      </Box>
                    </TableCell>
                    <TableCell>
                      {getResultChip(result)}
                    </TableCell>
                    <TableCell sx={{ color: 'rgba(255, 255, 255, 0.9)' }}>
                      {result.success ? 
                        `${((result.decision?.confidence || 0) * 100).toFixed(1)}%` : 
                        'N/A'
                      }
                    </TableCell>
                    <TableCell sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                      {(result.processing_time || 0).toFixed(2)}s
                    </TableCell>
                    <TableCell>
                      <Button
                        size="small"
                        startIcon={<Visibility />}
                        onClick={() => {
                          console.log('View details for:', result);
                          // You could implement a detailed view modal here
                        }}
                        sx={{
                          color: 'rgba(255, 255, 255, 0.8)',
                          '&:hover': {
                            color: '#00e5ff',
                            background: 'rgba(0, 229, 255, 0.1)',
                          }
                        }}
                      >
                        Details
                      </Button>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Box>
      )}
    </Box>
  );
};

export default BatchUpload;
