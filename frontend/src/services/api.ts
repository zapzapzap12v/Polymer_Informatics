// Mock API responses for presentation

interface PolymersResponse {
  smiles: string;
  predicted_capacitance: number;
  backbone: string;
}

interface PredictionResponse {
  polymer_id: string;
  predicted_value: number;
  confidence: number;
}

interface DiscoveryResponse {
  task_id: string;
  status: string;
  results: Array<{
    rank: number;
    smiles: string;
    predicted: number;
    error: number;
    material: string;
    thickness: number;
  }>;
  total_candidates: number;
  qualified_matches: number;
}

// Mock data generator
const generateMockPolymers = (count: number) => {
  const materials = ['PE', 'PP', 'PVC', 'PTFE', 'PVDF', 'PS', 'PMMA', 'PC', 'PET', 'PA6'];
  const results = [];
  for (let i = 0; i < count; i++) {
    results.push({
      rank: i + 1,
      smiles: `C${i}CC(C)C${i}`,
      predicted: 200 + (Math.random() - 0.5) * 20,
      error: Math.random() * 5,
      material: materials[Math.floor(Math.random() * materials.length)],
      thickness: 500 + Math.random() * 2500,
    });
  }
  return results.sort((a, b) => a.error - b.error).slice(0, 6);
};

class ApiService {
  // Polymer operations
  async generatePolymer(config: any): Promise<PolymersResponse> {
    // Simulate network delay
    await new Promise(resolve => setTimeout(resolve, 800));
    
    // For presentation, return mock but realistic data
    return {
      smiles: `${config.leftGroup}C${config.backbone}C${config.rightGroup}`,
      predicted_capacitance: 180 + Math.random() * 40,
      backbone: config.backbone,
    };
  }

  async validatePolymer(_smiles: string): Promise<{ valid: boolean; errors?: string[] }> {
    await new Promise(resolve => setTimeout(resolve, 300));
    return { valid: true };
  }

  // Model operations
  async getModels(): Promise<any[]> {
    await new Promise(resolve => setTimeout(resolve, 500));
    return [
      {
        id: 'ensemble',
        name: '3-Way Ensemble',
        r2: 0.9558,
        cv: 0.9073,
        rmse: 12.34,
        mae: 8.76,
      },
      {
        id: 'gnn',
        name: 'Graph Neural Network',
        r2: 0.9528,
        cv: 0.9301,
        rmse: 13.12,
        mae: 9.34,
      },
      {
        id: 'vae',
        name: 'Variational Autoencoder',
        r2: 0.8945,
        cv: 0.8712,
        rmse: 18.56,
        mae: 14.23,
      },
    ];
  }

  async predictPolymer(_smiles: string, _model: string): Promise<PredictionResponse> {
    await new Promise(resolve => setTimeout(resolve, 600));
    return {
      polymer_id: `polymer_${Date.now()}`,
      predicted_value: 180 + Math.random() * 40,
      confidence: 0.85 + Math.random() * 0.14,
    };
  }

  // Inverse design operations
  async searchPolymers(params: {
    targetCapacitance: number;
    librarySize: number;
    materials: string[];
    model: string;
  }): Promise<DiscoveryResponse> {
    // Simulate long search (2-3 seconds for dramatic effect)
    await new Promise(resolve => setTimeout(resolve, 2500));
    
    const results = generateMockPolymers(6);
    return {
      task_id: `task_${Date.now()}`,
      status: 'completed',
      results,
      total_candidates: params.librarySize,
      qualified_matches: Math.floor(params.librarySize * 0.005),
    };
  }

  async getSearchResults(taskId: string): Promise<DiscoveryResponse> {
    await new Promise(resolve => setTimeout(resolve, 300));
    return {
      task_id: taskId,
      status: 'completed',
      results: generateMockPolymers(6),
      total_candidates: 25000,
      qualified_matches: 127,
    };
  }

  // Export operations
  async exportResults(format: 'csv' | 'json' | 'pdf', results: any[]): Promise<Blob> {
    await new Promise(resolve => setTimeout(resolve, 800));
    
    let content = '';
    if (format === 'csv') {
      content = 'Rank,Polymer,Target,Predicted,Error,Material,Thickness\n';
      results.forEach((r, i) => {
        content += `${i + 1},Polymer_${i},200,${r.predicted.toFixed(2)},${r.error.toFixed(2)},${r.material},${r.thickness.toFixed(0)}\n`;
      });
      return new Blob([content], { type: 'text/csv' });
    } else if (format === 'json') {
      content = JSON.stringify(results, null, 2);
      return new Blob([content], { type: 'application/json' });
    } else {
      content = 'PDF Report - Polymer Discovery Results';
      return new Blob([content], { type: 'application/pdf' });
    }
  }

  // Metrics
  async getMetrics(): Promise<any> {
    await new Promise(resolve => setTimeout(resolve, 400));
    return {
      r2_score: 0.9558,
      accuracy: 95.3,
      simulations_success: 551,
      simulations_total: 1440,
      mse: 0.0125,
      rmse: 0.1118,
      mae: 0.0847,
    };
  }

  // Health check
  async health(): Promise<{ status: string }> {
    try {
      await new Promise(resolve => setTimeout(resolve, 200));
      return { status: 'ok' };
    } catch {
      return { status: 'error' };
    }
  }
}

export const apiService = new ApiService();
