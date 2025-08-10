import React, { useState, useEffect } from 'react';
import {
  Paper,
  Typography,
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Button,
  Chip,
  Alert,
  LinearProgress,
} from '@mui/material';
import {
  Refresh,
  Delete,
  Visibility,
  Schedule,
  CheckCircle,
  Error,
  HourglassEmpty,
} from '@mui/icons-material';
import { api } from '../services/api';

const JobStatus = ({ onError }) => {
  const [jobs, setJobs] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadJobs();
    const interval = setInterval(loadJobs, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const loadJobs = async () => {
    setLoading(true);
    try {
      const response = await api.listJobs();
      setJobs(response.data.jobs || []);
    } catch (error) {
      onError('Failed to load jobs');
    } finally {
      setLoading(false);
    }
  };

  const deleteJob = async (jobId) => {
    try {
      await api.deleteJob(jobId);
      setJobs(prev => prev.filter(job => job.job_id !== jobId));
    } catch (error) {
      onError('Failed to delete job');
    }
  };

  const getStatusIcon = (status) => {
  switch (status) {
    case 'completed':
      return <CheckCircle color="success" />;
    case 'failed':
      return <Error color="error" />;
    case 'processing':
      return <HourglassEmpty color="info" />;  // ✅ Fixed
    default:
      return <Schedule color="default" />;
  }
};


  const getStatusChip = (status) => {
    const colors = {
      pending: 'default',
      processing: 'info',
      completed: 'success',
      failed: 'error',
    };
    
    return (
      <Chip
        label={status.toUpperCase()}
        color={colors[status] || 'default'}
        size="small"
        icon={getStatusIcon(status)}
      />
    );
  };

  return (
    <Box>
      <Paper elevation={2} sx={{ p: 3 }}>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Typography variant="h5">Job Management</Typography>
          <Button
            variant="outlined"
            onClick={loadJobs}
            startIcon={<Refresh />}
            disabled={loading}
          >
            Refresh
          </Button>
        </Box>

        {loading && <LinearProgress sx={{ mb: 2 }} />}

        {jobs.length === 0 ? (
          <Alert severity="info">
            No active jobs. Jobs will appear here when you use async processing.
          </Alert>
        ) : (
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Job ID</TableCell>
                  <TableCell>File Name</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Created At</TableCell>
                  <TableCell>Completed At</TableCell>
                  <TableCell>Actions</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {jobs.map((job) => (
                  <TableRow key={job.job_id}>
                    <TableCell>
                      <Typography variant="body2" fontFamily="monospace">
                        {job.job_id.substring(0, 8)}...
                      </Typography>
                    </TableCell>
                    <TableCell>{job.file_name || 'Unknown'}</TableCell>
                    <TableCell>{getStatusChip(job.status)}</TableCell>
                    <TableCell>
                      {new Date(job.created_at).toLocaleString()}
                    </TableCell>
                    <TableCell>
                      {job.completed_at ? 
                        new Date(job.completed_at).toLocaleString() : 
                        '-'
                      }
                    </TableCell>
                    <TableCell>
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        {job.status === 'completed' && (
                          <Button
                            size="small"
                            startIcon={<Visibility />}
                            onClick={async () => {
                              try {
                                const response = await api.getJobStatus(job.job_id);
                                console.log('Job details:', response.data);
                                // You could open a modal with detailed results here
                              } catch (error) {
                                onError('Failed to get job details');
                              }
                            }}
                          >
                            View
                          </Button>
                        )}
                        {['completed', 'failed'].includes(job.status) && (
                          <Button
                            size="small"
                            color="error"
                            startIcon={<Delete />}
                            onClick={() => deleteJob(job.job_id)}
                          >
                            Delete
                          </Button>
                        )}
                      </Box>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Paper>
    </Box>
  );
};

export default JobStatus;
