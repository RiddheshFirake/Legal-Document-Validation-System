import React, { useState, useCallback } from 'react';
import {
  Typography,
  Button,
  Box,
  LinearProgress,
  FormControlLabel,
  Switch,
  TextField,
  Chip,
  Stack,
} from '@mui/material';
import {
  CloudUpload,
  Link as LinkIcon,
  Description,
} from '@mui/icons-material';
import { api } from '../services/api';

const DocumentUpload = ({ onValidationComplete, onError, loading, setLoading }) => {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadMode, setUploadMode] = useState('file'); // 'file' or 'url'
  const [documentUrl, setDocumentUrl] = useState('');
  const [options, setOptions] = useState({
    returnDetailed: false,
    asyncProcessing: false,
  });

  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setSelectedFile(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const validateFile = async () => {
    if (!selectedFile) {
      onError('Please select a file first');
      return;
    }

    setLoading(true);
    try {
      const response = await api.validateDocument(selectedFile, options);
      onValidationComplete(response.data);
      onError(null);
    } catch (error) {
      onError(error.response?.data?.detail || 'Validation failed');
    } finally {
      setLoading(false);
    }
  };

  const validateUrl = async () => {
    if (!documentUrl.trim()) {
      onError('Please enter a valid URL');
      return;
    }

    setLoading(true);
    try {
      const response = await api.validateUrl(documentUrl, options);
      onValidationComplete(response.data);
      onError(null);
    } catch (error) {
      onError(error.response?.data?.detail || 'URL validation failed');
    } finally {
      setLoading(false);
    }
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
        }}
      >
        Document Upload
      </Typography>

      {/* Upload Mode Toggle with Glass Effect */}
      <Box sx={{ mb: 3, textAlign: 'center' }}>
        <Button
          variant={uploadMode === 'file' ? 'contained' : 'outlined'}
          onClick={() => {
            setUploadMode('file');
            setDocumentUrl('');
          }}
          sx={{ 
            mr: 1,
            borderRadius: '16px',
            textTransform: 'none',
            fontWeight: 600,
            backdropFilter: 'blur(10px)',
            border: uploadMode === 'file' 
              ? '1px solid rgba(0, 229, 255, 0.5)' 
              : '1px solid rgba(255, 255, 255, 0.2)',
            background: uploadMode === 'file'
              ? 'linear-gradient(135deg, rgba(0, 229, 255, 0.2), rgba(24, 255, 255, 0.2))'
              : 'rgba(255, 255, 255, 0.05)',
            '&:hover': {
              background: uploadMode === 'file'
                ? 'linear-gradient(135deg, rgba(0, 229, 255, 0.35), rgba(24, 255, 255, 0.35))'
                : 'rgba(0, 229, 255, 0.1)',
              transform: 'translateY(-1px)',
            }
          }}
        >
          Upload File
        </Button>
        <Button
          variant={uploadMode === 'url' ? 'contained' : 'outlined'}
          onClick={() => {
            setUploadMode('url');
            setSelectedFile(null);
          }}
          startIcon={<LinkIcon />}
          sx={{ 
            borderRadius: '16px',
            textTransform: 'none',
            fontWeight: 600,
            backdropFilter: 'blur(10px)',
            border: uploadMode === 'url' 
              ? '1px solid rgba(0, 229, 255, 0.5)' 
              : '1px solid rgba(255, 255, 255, 0.2)',
            background: uploadMode === 'url'
              ? 'linear-gradient(135deg, rgba(0, 229, 255, 0.2), rgba(24, 255, 255, 0.2))'
              : 'rgba(255, 255, 255, 0.05)',
            '&:hover': {
              background: uploadMode === 'url'
                ? 'linear-gradient(135deg, rgba(0, 229, 255, 0.35), rgba(24, 255, 255, 0.35))'
                : 'rgba(0, 229, 255, 0.1)',
              transform: 'translateY(-1px)',
            }
          }}
        >
          From URL
        </Button>
      </Box>

      {uploadMode === 'file' ? (
        /* Enhanced File Upload Section with Glassmorphism */
        <Box>
          <Box
            className={`upload-zone ${dragActive ? 'drag-active' : ''}`}
            sx={{
              border: dragActive 
                ? '2px dashed rgba(0, 229, 255, 0.8)' 
                : '2px dashed rgba(255, 255, 255, 0.3)',
              borderRadius: '24px',
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
                transform: 'translateY(-4px)',
                '&::before': {
                  left: '100%',
                }
              },
            }}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => document.getElementById('file-input').click()}
          >
            <CloudUpload 
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
              {selectedFile ? 'File Selected' : 'Drag & Drop or Click to Upload'}
            </Typography>
            <Typography 
              variant="body2" 
              sx={{ 
                color: 'rgba(255, 255, 255, 0.7)',
                textShadow: '0 1px 2px rgba(0, 0, 0, 0.2)',
              }}
            >
              Supported formats: PDF, DOCX, JPG, PNG, TXT
            </Typography>
            <input
              id="file-input"
              type="file"
              accept=".pdf,.docx,.doc,.jpg,.jpeg,.png,.txt"
              onChange={handleFileChange}
              style={{ display: 'none' }}
            />
          </Box>

          {selectedFile && (
            <Box sx={{ mb: 3, textAlign: 'center' }}>
              <Chip
                icon={<Description />}
                label={`${selectedFile.name} (${formatFileSize(selectedFile.size)})`}
                onDelete={() => setSelectedFile(null)}
                sx={{
                  background: 'rgba(0, 229, 255, 0.2)',
                  color: '#ffffff',
                  backdropFilter: 'blur(10px)',
                  border: '1px solid rgba(0, 229, 255, 0.3)',
                  fontWeight: 600,
                  fontSize: '0.9rem',
                  padding: '8px 4px',
                  '& .MuiChip-deleteIcon': {
                    color: 'rgba(255, 255, 255, 0.7)',
                    '&:hover': {
                      color: '#ff6b6b',
                    }
                  }
                }}
              />
            </Box>
          )}
        </Box>
      ) : (
        /* Enhanced URL Upload Section */
        <Box sx={{ mb: 3 }}>
          <TextField
            fullWidth
            label="Document URL"
            value={documentUrl}
            onChange={(e) => setDocumentUrl(e.target.value)}
            placeholder="https://example.com/document.pdf"
            variant="outlined"
            InputProps={{
              startAdornment: <LinkIcon sx={{ mr: 1, color: 'rgba(255, 255, 255, 0.7)' }} />,
            }}
            sx={{
              '& .MuiOutlinedInput-root': {
                background: 'rgba(255, 255, 255, 0.05)',
                borderRadius: '16px',
                backdropFilter: 'blur(10px)',
                border: '1px solid rgba(255, 255, 255, 0.2)',
                '& fieldset': {
                  borderColor: 'rgba(255, 255, 255, 0.2)',
                },
                '&:hover fieldset': {
                  borderColor: 'rgba(0, 229, 255, 0.5)',
                },
                '&.Mui-focused fieldset': {
                  borderColor: '#00e5ff',
                  borderWidth: '2px',
                },
              },
              '& .MuiInputLabel-root': {
                color: 'rgba(255, 255, 255, 0.7)',
                '&.Mui-focused': {
                  color: '#00e5ff',
                },
              },
              '& .MuiInputBase-input': {
                color: '#ffffff',
              },
            }}
          />
        </Box>
      )}

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
        <Stack spacing={1}>
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
          <FormControlLabel
            control={
              <Switch
                checked={options.asyncProcessing}
                onChange={(e) =>
                  setOptions({ ...options, asyncProcessing: e.target.checked })
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
                Async Processing
              </Typography>
            }
          />
        </Stack>
      </Box>

      {/* Enhanced Validate Button */}
      <Button
        variant="contained"
        size="large"
        fullWidth
        onClick={uploadMode === 'file' ? validateFile : validateUrl}
        disabled={loading || (uploadMode === 'file' ? !selectedFile : !documentUrl.trim())}
        startIcon={<Description />}
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
        {loading ? 'Validating Document...' : 'Validate Document'}
      </Button>

      {/* Enhanced Loading Section */}
      {loading && (
        <Box sx={{ mt: 3 }}>
          <LinearProgress 
            sx={{
              background: 'rgba(255, 255, 255, 0.1)',
              borderRadius: '10px',
              backdropFilter: 'blur(10px)',
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
            🔍 Processing document... This may take a few seconds.
          </Typography>
        </Box>
      )}
    </Box>
  );
};

export default DocumentUpload;
