import React from 'react';
import {
  Typography,
  Box,
  Alert,
  Chip,
  Card,
  CardContent,
  Grid,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
} from '@mui/material';
import {
  CheckCircle,
  Error,
  Warning,
  Info,
  Gavel,
  Speed,
  Assessment,
  ExpandMore,
  Visibility,
  Rule,
  Psychology,
  AutoAwesome,
} from '@mui/icons-material';

const ValidationResults = ({ result, loading }) => {
  if (loading) {
    return (
      <Box sx={{ height: '100%' }}>
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
          Validation Results
        </Typography>
        <Box 
          sx={{ 
            textAlign: 'center', 
            py: 6,
            background: 'rgba(255, 255, 255, 0.05)',
            borderRadius: '20px',
            backdropFilter: 'blur(15px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
          }}
        >
          <AutoAwesome 
            sx={{ 
              fontSize: 48, 
              color: '#00e5ff', 
              mb: 3,
              filter: 'drop-shadow(0 4px 12px rgba(0, 229, 255, 0.5))',
              animation: 'pulse 2s ease-in-out infinite',
            }} 
          />
          <LinearProgress 
            sx={{ 
              mb: 3,
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
            sx={{ 
              color: 'rgba(255, 255, 255, 0.8)',
              fontWeight: 500,
              fontSize: '1.1rem',
              textShadow: '0 1px 2px rgba(0, 0, 0, 0.2)',
            }}
          >
            🔍 Analyzing document with AI...
          </Typography>
        </Box>
      </Box>
    );
  }

  if (!result) {
    return (
      <Box sx={{ height: '100%' }}>
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
          Validation Results
        </Typography>
        <Box 
          sx={{ 
            textAlign: 'center', 
            py: 6,
            background: 'rgba(255, 255, 255, 0.05)',
            borderRadius: '20px',
            backdropFilter: 'blur(15px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
          }}
        >
          <Info 
            sx={{ 
              fontSize: 64, 
              color: 'rgba(255, 255, 255, 0.4)', 
              mb: 3,
              filter: 'drop-shadow(0 4px 8px rgba(0, 0, 0, 0.3))',
            }} 
          />
          <Typography 
            sx={{ 
              color: 'rgba(255, 255, 255, 0.7)',
              fontSize: '1.1rem',
              fontWeight: 500,
            }}
          >
            📄 Upload a document to see validation results
          </Typography>
        </Box>
      </Box>
    );
  }

  const { result: validationData } = result;
  const decision = validationData?.decision;
  const isLegal = decision?.is_legal;
  const confidence = decision?.confidence || 0;

  const getConfidenceColor = (conf) => {
    if (conf >= 0.8) return 'success';
    if (conf >= 0.6) return 'warning';
    return 'error';
  };

  const getDecisionIcon = (legal, conf) => {
    if (legal && conf >= 0.7) return <CheckCircle color="success" />;
    if (!legal && conf >= 0.7) return <Error color="error" />;
    return <Warning color="warning" />;
  };

  return (
    <Box sx={{ height: '100%' }}>
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
        Validation Results
      </Typography>

      {!validationData?.success ? (
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
        >
          <Typography variant="subtitle1" sx={{ fontWeight: 600 }}>
            Validation Failed
          </Typography>
          <Typography variant="body2">
            {validationData?.error || 'Unknown error occurred'}
          </Typography>
        </Alert>
      ) : (
        <Box>
          {/* Enhanced Main Decision Card */}
          <Card 
            elevation={0}
            sx={{ 
              mb: 3, 
              background: isLegal 
                ? 'rgba(76, 175, 80, 0.15)' 
                : 'rgba(244, 67, 54, 0.15)',
              border: isLegal 
                ? '2px solid rgba(76, 175, 80, 0.4)' 
                : '2px solid rgba(244, 67, 54, 0.4)',
              borderRadius: '24px',
              backdropFilter: 'blur(20px)',
              boxShadow: isLegal
                ? '0 8px 32px rgba(76, 175, 80, 0.2)'
                : '0 8px 32px rgba(244, 67, 54, 0.2)',
              overflow: 'hidden',
              position: 'relative',
              '&::before': {
                content: '""',
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                height: '1px',
                background: `linear-gradient(90deg, transparent, ${
                  isLegal ? 'rgba(76, 175, 80, 0.6)' : 'rgba(244, 67, 54, 0.6)'
                }, transparent)`,
              }
            }}
          >
            <CardContent sx={{ p: 3 }}>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
                <Box sx={{ 
                  filter: isLegal 
                    ? 'drop-shadow(0 4px 12px rgba(76, 175, 80, 0.5))'
                    : 'drop-shadow(0 4px 12px rgba(244, 67, 54, 0.5))',
                }}>
                  {getDecisionIcon(isLegal, confidence)}
                </Box>
                <Typography 
                  variant="h4" 
                  sx={{ 
                    ml: 2, 
                    fontWeight: 'bold',
                    color: '#ffffff',
                    textShadow: '0 2px 10px rgba(0, 0, 0, 0.3)',
                    background: isLegal 
                      ? 'linear-gradient(45deg, #4caf50, #8bc34a)'
                      : 'linear-gradient(45deg, #f44336, #ff5722)',
                    backgroundClip: 'text',
                    WebkitBackgroundClip: 'text',
                    color: 'transparent',
                  }}
                >
                  {decision?.explanation?.decision || (isLegal ? 'LEGAL' : 'NOT LEGAL')}
                </Typography>
              </Box>

              <Grid container spacing={3}>
                <Grid item xs={12} sm={4}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography 
                      variant="h6" 
                      sx={{ 
                        color: 'rgba(255, 255, 255, 0.8)',
                        fontWeight: 600,
                        mb: 1,
                      }}
                    >
                      Confidence
                    </Typography>
                    <Chip
                      label={`${(confidence * 100).toFixed(1)}%`}
                      color={getConfidenceColor(confidence)}
                      size="large"
                      sx={{ 
                        fontSize: '1.2rem', 
                        fontWeight: 700,
                        py: 1,
                        background: `rgba(${getConfidenceColor(confidence) === 'success' ? '76, 175, 80' : getConfidenceColor(confidence) === 'warning' ? '255, 152, 0' : '244, 67, 54'}, 0.2)`,
                        backdropFilter: 'blur(10px)',
                        border: `1px solid rgba(${getConfidenceColor(confidence) === 'success' ? '76, 175, 80' : getConfidenceColor(confidence) === 'warning' ? '255, 152, 0' : '244, 67, 54'}, 0.4)`,
                        color: '#ffffff',
                      }}
                    />
                  </Box>
                </Grid>

                <Grid item xs={12} sm={4}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography 
                      variant="h6" 
                      sx={{ 
                        color: 'rgba(255, 255, 255, 0.8)',
                        fontWeight: 600,
                        mb: 1,
                      }}
                    >
                      Risk Level
                    </Typography>
                    <Chip
                      label={decision?.risk_assessment?.risk_level || 'Unknown'}
                      color={
                        decision?.risk_assessment?.risk_level === 'LOW' ? 'success' :
                        decision?.risk_assessment?.risk_level === 'MEDIUM' ? 'warning' : 'error'
                      }
                      size="large"
                      sx={{ 
                        fontWeight: 700,
                        background: `rgba(${decision?.risk_assessment?.risk_level === 'LOW' ? '76, 175, 80' : decision?.risk_assessment?.risk_level === 'MEDIUM' ? '255, 152, 0' : '244, 67, 54'}, 0.2)`,
                        backdropFilter: 'blur(10px)',
                        border: `1px solid rgba(${decision?.risk_assessment?.risk_level === 'LOW' ? '76, 175, 80' : decision?.risk_assessment?.risk_level === 'MEDIUM' ? '255, 152, 0' : '244, 67, 54'}, 0.4)`,
                        color: '#ffffff',
                      }}
                    />
                  </Box>
                </Grid>

                <Grid item xs={12} sm={4}>
                  <Box sx={{ textAlign: 'center' }}>
                    <Typography 
                      variant="h6" 
                      sx={{ 
                        color: 'rgba(255, 255, 255, 0.8)',
                        fontWeight: 600,
                        mb: 1,
                      }}
                    >
                      Processing Time
                    </Typography>
                    <Chip
                      icon={<Speed sx={{ filter: 'drop-shadow(0 2px 4px rgba(0, 229, 255, 0.5))' }} />}
                      label={`${(validationData.processing_time || 0).toFixed(2)}s`}
                      size="large"
                      sx={{ 
                        fontWeight: 700,
                        background: 'rgba(0, 229, 255, 0.2)',
                        backdropFilter: 'blur(10px)',
                        border: '1px solid rgba(0, 229, 255, 0.4)',
                        color: '#ffffff',
                      }}
                    />
                  </Box>
                </Grid>
              </Grid>

              {decision?.explanation?.summary && (
                <Alert 
                  severity="info" 
                  sx={{ 
                    mt: 3,
                    background: 'rgba(33, 150, 243, 0.15)',
                    backdropFilter: 'blur(15px)',
                    border: '1px solid rgba(33, 150, 243, 0.3)',
                    borderRadius: '16px',
                    color: '#ffffff',
                    '& .MuiAlert-icon': {
                      color: '#64b5f6',
                    }
                  }}
                >
                  {decision.explanation.summary}
                </Alert>
              )}
            </CardContent>
          </Card>

          {/* Enhanced Detailed Analysis Accordion */}
          <Accordion 
            defaultExpanded
            sx={{
              background: 'rgba(255, 255, 255, 0.05)',
              backdropFilter: 'blur(15px)',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              borderRadius: '16px',
              mb: 2,
              '&:before': { display: 'none' },
              boxShadow: '0 8px 32px rgba(31, 38, 135, 0.37)',
            }}
          >
            <AccordionSummary 
              expandIcon={<ExpandMore sx={{ color: 'rgba(255, 255, 255, 0.8)' }} />}
              sx={{ 
                background: 'rgba(255, 255, 255, 0.05)',
                borderRadius: '16px 16px 0 0',
                '& .MuiAccordionSummary-content': {
                  alignItems: 'center',
                }
              }}
            >
              <Assessment sx={{ mr: 1, color: '#00e5ff', filter: 'drop-shadow(0 2px 4px rgba(0, 229, 255, 0.5))' }} />
              <Typography variant="h6" sx={{ color: '#ffffff', fontWeight: 600 }}>
                Detailed Analysis
              </Typography>
            </AccordionSummary>
            <AccordionDetails sx={{ background: 'rgba(255, 255, 255, 0.02)' }}>
              <Typography 
                variant="subtitle1" 
                gutterBottom 
                sx={{ 
                  color: 'rgba(255, 255, 255, 0.9)',
                  fontWeight: 600,
                  mb: 3,
                }}
              >
                Component Analysis
              </Typography>
              {decision?.component_scores && Object.entries(decision.component_scores).map(([component, data]) => (
                <Box 
                  key={component} 
                  sx={{ 
                    mb: 3,
                    p: 2,
                    background: 'rgba(255, 255, 255, 0.05)',
                    borderRadius: '12px',
                    border: '1px solid rgba(255, 255, 255, 0.1)',
                  }}
                >
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Box sx={{ display: 'flex', alignItems: 'center' }}>
                      {component === 'nlp' && <Psychology sx={{ mr: 1, color: '#9c27b0' }} />}
                      {component === 'vision' && <Visibility sx={{ mr: 1, color: '#2196f3' }} />}
                      {component === 'rules' && <Rule sx={{ mr: 1, color: '#ff9800' }} />}
                      <Typography 
                        variant="body1" 
                        sx={{ 
                          fontWeight: 600,
                          color: '#ffffff',
                          textTransform: 'capitalize',
                        }}
                      >
                        {component} Analysis
                      </Typography>
                    </Box>
                    <Chip
                      label={data.available ? 'Active' : 'Unavailable'}
                      color={data.available ? 'success' : 'default'}
                      size="small"
                      sx={{
                        background: data.available 
                          ? 'rgba(76, 175, 80, 0.2)' 
                          : 'rgba(255, 255, 255, 0.1)',
                        backdropFilter: 'blur(10px)',
                        border: data.available 
                          ? '1px solid rgba(76, 175, 80, 0.3)' 
                          : '1px solid rgba(255, 255, 255, 0.2)',
                        color: '#ffffff',
                        fontWeight: 600,
                      }}
                    />
                  </Box>
                  {data.available && (
                    <Box sx={{ mb: 2 }}>
                      <LinearProgress
                        variant="determinate"
                        value={(data.score || 0) * 100}
                        sx={{ 
                          height: 12, 
                          borderRadius: 6,
                          background: 'rgba(255, 255, 255, 0.1)',
                          '& .MuiLinearProgress-bar': {
                            background: data.score > 0.5 
                              ? 'linear-gradient(90deg, #4caf50, #8bc34a)'
                              : 'linear-gradient(90deg, #f44336, #ff5722)',
                            borderRadius: 6,
                          }
                        }}
                      />
                    </Box>
                  )}
                  <Grid container spacing={2}>
                    <Grid item xs={6}>
                      <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                        Score: <span style={{ color: '#ffffff', fontWeight: 600 }}>
                          {((data.score || 0) * 100).toFixed(1)}%
                        </span>
                      </Typography>
                    </Grid>
                    <Grid item xs={6}>
                      <Typography variant="body2" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                        Confidence: <span style={{ color: '#ffffff', fontWeight: 600 }}>
                          {((data.confidence || 0) * 100).toFixed(1)}%
                        </span>
                      </Typography>
                    </Grid>
                  </Grid>
                </Box>
              ))}
            </AccordionDetails>
          </Accordion>

          {/* Enhanced Key Factors Accordion */}
          {decision?.explanation?.key_factors && decision.explanation.key_factors.length > 0 && (
            <Accordion 
              sx={{
                background: 'rgba(255, 255, 255, 0.05)',
                backdropFilter: 'blur(15px)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '16px',
                mb: 2,
                '&:before': { display: 'none' },
              }}
            >
              <AccordionSummary 
                expandIcon={<ExpandMore sx={{ color: 'rgba(255, 255, 255, 0.8)' }} />}
                sx={{ background: 'rgba(255, 255, 255, 0.05)' }}
              >
                <Gavel sx={{ mr: 1, color: '#ffb74d', filter: 'drop-shadow(0 2px 4px rgba(255, 183, 77, 0.5))' }} />
                <Typography variant="h6" sx={{ color: '#ffffff', fontWeight: 600 }}>
                  Key Factors
                </Typography>
              </AccordionSummary>
              <AccordionDetails sx={{ background: 'rgba(255, 255, 255, 0.02)' }}>
                <List>
                  {decision.explanation.key_factors.map((factor, index) => (
                    <ListItem 
                      key={index}
                      sx={{
                        background: 'rgba(255, 255, 255, 0.05)',
                        borderRadius: '8px',
                        mb: 1,
                        border: '1px solid rgba(255, 255, 255, 0.05)',
                      }}
                    >
                      <ListItemIcon>
                        <Info sx={{ color: '#64b5f6' }} />
                      </ListItemIcon>
                      <ListItemText 
                        primary={factor}
                        primaryTypographyProps={{
                          sx: { color: 'rgba(255, 255, 255, 0.9)', fontWeight: 500 }
                        }}
                      />
                    </ListItem>
                  ))}
                </List>
              </AccordionDetails>
            </Accordion>
          )}

          {/* Enhanced Recommendations Accordion */}
          {decision?.recommendations && decision.recommendations.length > 0 && (
            <Accordion 
              sx={{
                background: 'rgba(255, 255, 255, 0.05)',
                backdropFilter: 'blur(15px)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '16px',
                mb: 2,
                '&:before': { display: 'none' },
              }}
            >
              <AccordionSummary 
                expandIcon={<ExpandMore sx={{ color: 'rgba(255, 255, 255, 0.8)' }} />}
                sx={{ background: 'rgba(255, 255, 255, 0.05)' }}
              >
                <Warning sx={{ mr: 1, color: '#ffab40', filter: 'drop-shadow(0 2px 4px rgba(255, 171, 64, 0.5))' }} />
                <Typography variant="h6" sx={{ color: '#ffffff', fontWeight: 600 }}>
                  Recommendations
                </Typography>
              </AccordionSummary>
              <AccordionDetails sx={{ background: 'rgba(255, 255, 255, 0.02)' }}>
                <List>
                  {decision.recommendations.map((recommendation, index) => (
                    <ListItem 
                      key={index}
                      sx={{
                        background: 'rgba(255, 152, 0, 0.1)',
                        borderRadius: '8px',
                        mb: 1,
                        border: '1px solid rgba(255, 152, 0, 0.2)',
                      }}
                    >
                      <ListItemIcon>
                        <Warning sx={{ color: '#ffab40' }} />
                      </ListItemIcon>
                      <ListItemText 
                        primary={recommendation}
                        primaryTypographyProps={{
                          sx: { color: 'rgba(255, 255, 255, 0.9)', fontWeight: 500 }
                        }}
                      />
                    </ListItem>
                  ))}
                </List>
              </AccordionDetails>
            </Accordion>
          )}

          {/* Enhanced Risk Assessment Accordion */}
          {decision?.risk_assessment && (
            <Accordion 
              sx={{
                background: 'rgba(255, 255, 255, 0.05)',
                backdropFilter: 'blur(15px)',
                border: '1px solid rgba(255, 255, 255, 0.1)',
                borderRadius: '16px',
                '&:before': { display: 'none' },
              }}
            >
              <AccordionSummary 
                expandIcon={<ExpandMore sx={{ color: 'rgba(255, 255, 255, 0.8)' }} />}
                sx={{ background: 'rgba(255, 255, 255, 0.05)' }}
              >
                <Error sx={{ mr: 1, color: '#ef5350', filter: 'drop-shadow(0 2px 4px rgba(239, 83, 80, 0.5))' }} />
                <Typography variant="h6" sx={{ color: '#ffffff', fontWeight: 600 }}>
                  Risk Assessment
                </Typography>
              </AccordionSummary>
              <AccordionDetails sx={{ background: 'rgba(255, 255, 255, 0.02)' }}>
                <Grid container spacing={3}>
                  <Grid item xs={12} sm={6}>
                    <Typography 
                      variant="body2" 
                      sx={{ 
                        color: 'rgba(255, 255, 255, 0.7)',
                        fontWeight: 600,
                        mb: 1,
                      }}
                    >
                      Risk Score
                    </Typography>
                    <LinearProgress
                      variant="determinate"
                      value={(decision.risk_assessment.risk_score || 0) * 100}
                      sx={{ 
                        height: 12, 
                        borderRadius: 6, 
                        mb: 2,
                        background: 'rgba(255, 255, 255, 0.1)',
                        '& .MuiLinearProgress-bar': {
                          background: 'linear-gradient(90deg, #f44336, #ff5722)',
                          borderRadius: 6,
                        }
                      }}
                    />
                    <Typography variant="body1" sx={{ color: '#ffffff', fontWeight: 600 }}>
                      {((decision.risk_assessment.risk_score || 0) * 100).toFixed(1)}%
                    </Typography>
                  </Grid>
                  <Grid item xs={12} sm={6}>
                    <Typography 
                      variant="body2" 
                      sx={{ 
                        color: 'rgba(255, 255, 255, 0.7)',
                        fontWeight: 600,
                        mb: 1,
                      }}
                    >
                      Human Review Required
                    </Typography>
                    <Chip
                      label={decision.risk_assessment.requires_human_review ? 'YES' : 'NO'}
                      color={decision.risk_assessment.requires_human_review ? 'error' : 'success'}
                      sx={{
                        fontWeight: 700,
                        background: decision.risk_assessment.requires_human_review 
                          ? 'rgba(244, 67, 54, 0.2)' 
                          : 'rgba(76, 175, 80, 0.2)',
                        backdropFilter: 'blur(10px)',
                        border: decision.risk_assessment.requires_human_review 
                          ? '1px solid rgba(244, 67, 54, 0.4)' 
                          : '1px solid rgba(76, 175, 80, 0.4)',
                        color: '#ffffff',
                      }}
                    />
                  </Grid>
                </Grid>

                {decision.risk_assessment.risk_factors && decision.risk_assessment.risk_factors.length > 0 && (
                  <Box sx={{ mt: 3 }}>
                    <Typography 
                      variant="subtitle2" 
                      gutterBottom
                      sx={{ 
                        color: 'rgba(255, 255, 255, 0.9)',
                        fontWeight: 600,
                        mb: 2,
                      }}
                    >
                      Risk Factors:
                    </Typography>
                    <List dense>
                      {decision.risk_assessment.risk_factors.map((factor, index) => (
                        <ListItem 
                          key={index}
                          sx={{
                            background: 'rgba(244, 67, 54, 0.1)',
                            borderRadius: '8px',
                            mb: 1,
                            border: '1px solid rgba(244, 67, 54, 0.2)',
                          }}
                        >
                          <ListItemIcon>
                            <Error sx={{ color: '#ef5350', fontSize: 'small' }} />
                          </ListItemIcon>
                          <ListItemText 
                            primary={factor}
                            primaryTypographyProps={{
                              sx: { color: 'rgba(255, 255, 255, 0.9)', fontWeight: 500 }
                            }}
                          />
                        </ListItem>
                      ))}
                    </List>
                  </Box>
                )}
              </AccordionDetails>
            </Accordion>
          )}
        </Box>
      )}
    </Box>
  );
};

export default ValidationResults;
