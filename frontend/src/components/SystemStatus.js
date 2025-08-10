import React, { useState, useEffect } from 'react';
import {
  Paper,
  Typography,
  Box,
  Grid,
  Card,
  CardContent,
  Button,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  LinearProgress,
} from '@mui/material';
import {
  Refresh,
  CheckCircle,
  Error,
  Computer,
  Psychology,
  Visibility,
  Rule,
} from '@mui/icons-material';
import { api } from '../services/api';

const SystemStatus = ({ systemHealth, onRefresh, onError }) => {
  const [pipelineStatus, setPipelineStatus] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadPipelineStatus();
  }, []);

  const loadPipelineStatus = async () => {
    setLoading(true);
    try {
      const response = await api.getPipelineStatus();
      setPipelineStatus(response.data);
    } catch (error) {
      onError('Failed to load pipeline status');
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = () => {
    onRefresh();
    loadPipelineStatus();
  };

  const getComponentIcon = (component) => {
    switch (component) {
      case 'nlp_classifier':
      case 'text_preprocessor':
      case 'vectorizer':
        return <Psychology />;
      case 'vision_model':
        return <Visibility />;
      case 'file_detector':
      case 'pdf_extractor':
      case 'ocr_processor':
        return <Computer />;
      default:
        return <Rule />;
    }
  };

  const componentDescriptions = {
    file_detector: 'File Type Detection',
    ocr_processor: 'OCR Text Extraction',
    pdf_extractor: 'PDF Processing',
    text_preprocessor: 'Text Preprocessing',
    vectorizer: 'Text Vectorization',
    nlp_classifier: 'NLP Classification Model',
    vision_model: 'Computer Vision Model',
    mrz_validator: 'MRZ Code Validation',
    clause_validator: 'Legal Clause Detection',
    rules_validator: 'Rule-based Validation',
    signature_validator: 'Signature Verification',
  };

  return (
    <Box>
      <Paper elevation={2} sx={{ p: 3, mb: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h5">System Status</Typography>
          <Button
            variant="outlined"
            onClick={handleRefresh}
            startIcon={<Refresh />}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>

        <Grid container spacing={3}>
          {/* Overall Health */}
          <Grid item xs={12} md={4}>
            <Card elevation={2}>
              <CardContent sx={{ textAlign: 'center' }}>
                <Typography variant="h6" gutterBottom>
                  Overall Health
                </Typography>
                {systemHealth ? (
                  <Chip
                    icon={systemHealth.status === 'healthy' ? <CheckCircle /> : <Error />}
                    label={systemHealth.status === 'healthy' ? 'Healthy' : 'Error'}
                    color={systemHealth.status === 'healthy' ? 'success' : 'error'}
                    size="large"
                  />
                ) : (
                  <Chip label="Unknown" color="default" size="large" />
                )}
                {systemHealth?.timestamp && (
                  <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                    Last checked: {new Date(systemHealth.timestamp).toLocaleTimeString()}
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>

          {/* Component Status Summary */}
          <Grid item xs={12} md={8}>
            <Card elevation={2}>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  Component Summary
                </Typography>
                {loading ? (
                  <LinearProgress />
                ) : pipelineStatus ? (
                  <Grid container spacing={2}>
                    {Object.entries(pipelineStatus)
                      .filter(([key]) => key !== 'timestamp')
                      .map(([component, status]) => (
                      <Grid item xs={6} sm={4} md={3} key={component}>
                        <Box sx={{ textAlign: 'center', p: 1 }}>
                          {getComponentIcon(component)}
                          <Typography variant="body2" noWrap>
                            {componentDescriptions[component] || component}
                          </Typography>
                          <Chip
                            label={status ? 'Active' : 'Inactive'}
                            color={status ? 'success' : 'default'}
                            size="small"
                          />
                        </Box>
                      </Grid>
                    ))}
                  </Grid>
                ) : (
                  <Typography color="textSecondary">
                    Failed to load component status
                  </Typography>
                )}
              </CardContent>
            </Card>
          </Grid>
        </Grid>
      </Paper>

      {/* Detailed Component Status */}
      {pipelineStatus && (
        <Paper elevation={2} sx={{ p: 3 }}>
          <Typography variant="h6" gutterBottom>
            Component Details
          </Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Component</TableCell>
                  <TableCell>Description</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Category</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {Object.entries(pipelineStatus)
                  .filter(([key]) => key !== 'timestamp')
                  .map(([component, status]) => (
                  <TableRow key={component}>
                    <TableCell>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        {getComponentIcon(component)}
                        <Typography sx={{ ml: 1 }}>
                          {component.replace('_', ' ').toUpperCase()}
                        </Typography>
                      </Box>
                    </TableCell>
                    <TableCell>
                      {componentDescriptions[component] || 'System component'}
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={status ? 'Active' : 'Inactive'}
                        color={status ? 'success' : 'default'}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Typography variant="body2" color="textSecondary">
                        {component.includes('nlp') || component.includes('text') || component.includes('vector') ? 'NLP' :
                         component.includes('vision') ? 'Computer Vision' :
                         component.includes('validator') ? 'Validation' :
                         'Infrastructure'}
                      </Typography>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Paper>
      )}
    </Box>
  );
};

export default SystemStatus;
