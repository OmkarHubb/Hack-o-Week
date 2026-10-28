import React from 'react';
import { LabProvider, useLab } from './context/LabContext';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { Module1PyTorchBasics } from './pages/Module1PyTorchBasics';
import { Module2CNN } from './pages/Module2CNN';
import { Module3RNN } from './pages/Module3RNN';
import { Module4Regularization } from './pages/Module4Regularization';
import { Module5TransferLearning } from './pages/Module5TransferLearning';
import { Module6Transformers } from './pages/Module6Transformers';
import { Module7LLMApis } from './pages/Module7LLMApis';

const ModuleRouter: React.FC = () => {
  const { activeModule } = useLab();

  switch (activeModule) {
    case 1:
      return <Module1PyTorchBasics />;
    case 2:
      return <Module2CNN />;
    case 3:
      return <Module3RNN />;
    case 4:
      return <Module4Regularization />;
    case 5:
      return <Module5TransferLearning />;
    case 6:
      return <Module6Transformers />;
    case 7:
      return <Module7LLMApis />;
    case 0:
    default:
      return <Dashboard />;
  }
};

export function App() {
  return (
    <LabProvider>
      <Layout>
        <ModuleRouter />
      </Layout>
    </LabProvider>
  );
}

export default App;
