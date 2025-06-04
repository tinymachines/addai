import init, { DataProcessor } from '../pkg/eeg_monitor_wasm.js';
import { Chart, registerables } from 'chart.js';
import 'chartjs-adapter-date-fns';

Chart.register(...registerables);

interface EEGData {
  timestamp: string;
  signal_quality: number;
  attention: number;
  meditation: number;
  eeg_power?: {
    delta: number;
    theta: number;
    low_alpha: number;
    high_alpha: number;
    low_beta: number;
    high_beta: number;
    low_gamma: number;
    mid_gamma: number;
  };
}

class EEGMonitor {
  private dataProcessor: DataProcessor | null = null;
  private websocket: WebSocket | null = null;
  private attentionChart: Chart | null = null;
  private meditationChart: Chart | null = null;
  private eegChart: Chart | null = null;
  private isConnected = false;

  async init() {
    try {
      // Initialize WASM module
      await init();
      this.dataProcessor = new DataProcessor(100); // Keep 100 data points

      // Hide loading indicator
      const loadingElement = document.getElementById('loading');
      if (loadingElement) {
        loadingElement.style.display = 'none';
      }

      // Setup UI
      this.setupCharts();
      this.setupConnection();
      this.updateConnectionStatus();
    } catch (error) {
      console.error('Failed to initialize WASM module:', error);
      const loadingElement = document.getElementById('loading');
      if (loadingElement) {
        loadingElement.innerHTML = '<p style="color: red;">Failed to load WASM module. Please refresh the page.</p>';
      }
    }
  }

  private setupCharts() {
    // Attention/Meditation Line Charts
    const attentionCtx = document.getElementById('attentionChart') as HTMLCanvasElement;
    this.attentionChart = new Chart(attentionCtx, {
      type: 'line',
      data: {
        datasets: [{
          label: 'Attention',
          data: [],
          borderColor: 'rgb(75, 192, 192)',
          backgroundColor: 'rgba(75, 192, 192, 0.2)',
          tension: 0.1
        }]
      },
      options: {
        responsive: true,
        scales: {
          x: {
            type: 'time',
            time: {
              displayFormats: {
                second: 'HH:mm:ss'
              }
            }
          },
          y: {
            beginAtZero: true,
            max: 100
          }
        },
        plugins: {
          title: {
            display: true,
            text: 'Attention Level'
          }
        }
      }
    });

    const meditationCtx = document.getElementById('meditationChart') as HTMLCanvasElement;
    this.meditationChart = new Chart(meditationCtx, {
      type: 'line',
      data: {
        datasets: [{
          label: 'Meditation',
          data: [],
          borderColor: 'rgb(153, 102, 255)',
          backgroundColor: 'rgba(153, 102, 255, 0.2)',
          tension: 0.1
        }]
      },
      options: {
        responsive: true,
        scales: {
          x: {
            type: 'time',
            time: {
              displayFormats: {
                second: 'HH:mm:ss'
              }
            }
          },
          y: {
            beginAtZero: true,
            max: 100
          }
        },
        plugins: {
          title: {
            display: true,
            text: 'Meditation Level'
          }
        }
      }
    });

    // EEG Power Bands Chart
    const eegCtx = document.getElementById('eegChart') as HTMLCanvasElement;
    this.eegChart = new Chart(eegCtx, {
      type: 'bar',
      data: {
        labels: ['Delta', 'Theta', 'Low Alpha', 'High Alpha', 'Low Beta', 'High Beta', 'Low Gamma', 'Mid Gamma'],
        datasets: [{
          label: 'EEG Power',
          data: [0, 0, 0, 0, 0, 0, 0, 0],
          backgroundColor: [
            'rgba(255, 99, 132, 0.8)',
            'rgba(54, 162, 235, 0.8)',
            'rgba(255, 205, 86, 0.8)',
            'rgba(75, 192, 192, 0.8)',
            'rgba(153, 102, 255, 0.8)',
            'rgba(255, 159, 64, 0.8)',
            'rgba(199, 199, 199, 0.8)',
            'rgba(83, 102, 255, 0.8)'
          ]
        }]
      },
      options: {
        responsive: true,
        scales: {
          y: {
            beginAtZero: true,
            type: 'logarithmic'
          }
        },
        plugins: {
          title: {
            display: true,
            text: 'EEG Power Bands'
          }
        }
      }
    });
  }

  private setupConnection() {
    const connectBtn = document.getElementById('connectBtn') as HTMLButtonElement;
    connectBtn.addEventListener('click', () => {
      if (this.isConnected) {
        this.disconnect();
      } else {
        this.connect();
      }
    });
  }

  private connect() {
    // Use relative URL to work with Vite proxy in development
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;
    console.log('Connecting to WebSocket:', wsUrl);
    this.websocket = new WebSocket(wsUrl);

    this.websocket.onopen = () => {
      console.log('WebSocket connected');
      this.isConnected = true;
      this.updateConnectionStatus();
    };

    this.websocket.onmessage = (event) => {
      try {
        const data: EEGData = JSON.parse(event.data);
        this.handleNewData(data);
      } catch (error) {
        console.error('Error parsing WebSocket data:', error);
      }
    };

    this.websocket.onclose = () => {
      console.log('WebSocket disconnected');
      this.isConnected = false;
      this.updateConnectionStatus();
    };

    this.websocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      console.error('Make sure the backend API is running on port 8000');
      this.isConnected = false;
      this.updateConnectionStatus();
      
      // Update status to show error
      const statusElement = document.getElementById('connectionStatus');
      if (statusElement) {
        statusElement.textContent = 'Connection Error';
      }
    };
  }

  private disconnect() {
    if (this.websocket) {
      this.websocket.close();
      this.websocket = null;
    }
    this.isConnected = false;
    this.updateConnectionStatus();
  }

  private handleNewData(data: EEGData) {
    if (!this.dataProcessor) return;

    // Add data to WASM processor
    this.dataProcessor.add_data(data);

    // Update charts
    this.updateAttentionChart(data);
    this.updateMeditationChart(data);
    if (data.eeg_power) {
      this.updateEEGChart(data.eeg_power);
    }

    // Update status indicators
    this.updateStatusIndicators(data);
  }

  private updateAttentionChart(data: EEGData) {
    if (!this.attentionChart) return;

    const dataset = this.attentionChart.data.datasets[0];
    dataset.data.push({
      x: new Date(data.timestamp),
      y: data.attention
    });

    // Keep only last 50 points
    if (dataset.data.length > 50) {
      dataset.data.shift();
    }

    this.attentionChart.update('none');
  }

  private updateMeditationChart(data: EEGData) {
    if (!this.meditationChart) return;

    const dataset = this.meditationChart.data.datasets[0];
    dataset.data.push({
      x: new Date(data.timestamp),
      y: data.meditation
    });

    // Keep only last 50 points
    if (dataset.data.length > 50) {
      dataset.data.shift();
    }

    this.meditationChart.update('none');
  }

  private updateEEGChart(eegPower: NonNullable<EEGData['eeg_power']>) {
    if (!this.eegChart) return;

    const dataset = this.eegChart.data.datasets[0];
    dataset.data = [
      eegPower.delta,
      eegPower.theta,
      eegPower.low_alpha,
      eegPower.high_alpha,
      eegPower.low_beta,
      eegPower.high_beta,
      eegPower.low_gamma,
      eegPower.mid_gamma
    ];

    this.eegChart.update('none');
  }

  private updateStatusIndicators(data: EEGData) {
    // Signal quality indicator
    const signalElement = document.getElementById('signalQuality');
    if (signalElement) {
      const quality = data.signal_quality === 0 ? 'Good' : 
                     data.signal_quality < 100 ? 'Fair' : 'Poor';
      signalElement.textContent = `Signal: ${quality} (${data.signal_quality})`;
      signalElement.className = `status ${quality.toLowerCase()}`;
    }

    // Current values
    const attentionElement = document.getElementById('currentAttention');
    if (attentionElement) {
      attentionElement.textContent = `Attention: ${data.attention}%`;
    }

    const meditationElement = document.getElementById('currentMeditation');
    if (meditationElement) {
      meditationElement.textContent = `Meditation: ${data.meditation}%`;
    }

    // Averages using WASM processor
    if (this.dataProcessor) {
      const attentionAvg = this.dataProcessor.calculate_attention_average(10);
      const meditationAvg = this.dataProcessor.calculate_meditation_average(10);
      
      const attentionAvgElement = document.getElementById('attentionAverage');
      if (attentionAvgElement) {
        attentionAvgElement.textContent = `Avg: ${attentionAvg.toFixed(1)}%`;
      }

      const meditationAvgElement = document.getElementById('meditationAverage');
      if (meditationAvgElement) {
        meditationAvgElement.textContent = `Avg: ${meditationAvg.toFixed(1)}%`;
      }
    }
  }

  private updateConnectionStatus() {
    const statusElement = document.getElementById('connectionStatus');
    const connectBtn = document.getElementById('connectBtn') as HTMLButtonElement;
    
    if (statusElement) {
      statusElement.textContent = this.isConnected ? 'Connected' : 'Disconnected';
      statusElement.className = `status ${this.isConnected ? 'good' : 'poor'}`;
    }
    
    if (connectBtn) {
      connectBtn.textContent = this.isConnected ? 'Disconnect' : 'Connect';
    }
  }
}

// Initialize the application
const monitor = new EEGMonitor();
monitor.init().catch(console.error);