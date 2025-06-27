"""
Test Pipeline Stages
CI-compatible tests for the stage-based pipeline architecture
"""

import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

# Set CI environment for testing
os.environ['CI'] = 'true'

from src.pipeline.stage_manager import StageManager
from src.pipeline.base_stage import BaseStage, StageResult


class TestStageManager:
    """Test the StageManager class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.state_dir = self.temp_dir / ".pipeline_state"
        self.manager = StageManager(
            project_root=self.temp_dir,
            state_dir=self.state_dir
        )
    
    def test_stage_manager_initialization(self):
        """Test stage manager initializes correctly"""
        assert self.manager.project_root == self.temp_dir
        assert self.manager.state_dir == self.state_dir
        assert self.manager.is_ci == True
        assert len(self.manager.stages) > 0
        assert 'preparation' in self.manager.stages
    
    def test_list_stages(self):
        """Test listing all stages"""
        stages = self.manager.list_stages()
        assert isinstance(stages, list)
        assert len(stages) > 0
        
        # Check stage structure
        stage = stages[0]
        assert 'name' in stage
        assert 'description' in stage
        assert 'environment' in stage
        assert 'dependencies' in stage
    
    def test_get_stage_status(self):
        """Test getting status of a specific stage"""
        status = self.manager.get_stage_status('preparation')
        assert status is not None
        assert status['name'] == 'preparation'
        assert 'completed' in status
        assert 'environment' in status
    
    def test_get_pipeline_status(self):
        """Test getting overall pipeline status"""
        status = self.manager.get_pipeline_status()
        assert 'total_stages' in status
        assert 'completed_stages' in status
        assert 'overall_progress' in status
        assert 'execution_order' in status
        assert 'ci_mode' in status
        assert status['ci_mode'] == True
    
    def test_run_single_stage_ci_mode(self):
        """Test running a single stage in CI mode"""
        result = self.manager.run_single_stage('preparation')
        assert isinstance(result, StageResult)
        assert result.success == True
        assert 'ci_mock' in result.data.get('status', '')
    
    def test_run_stages_ci_mode(self):
        """Test running all stages in CI mode"""
        summary = self.manager.run_stages()
        assert summary['success'] == True
        assert summary['ci_mode'] == True
        assert summary['stages_run'] > 0
        assert 'results' in summary
    
    def test_reset_pipeline(self):
        """Test resetting pipeline state"""
        # This should not raise an error
        self.manager.reset_pipeline()
        
        # Test resetting specific stages
        self.manager.reset_pipeline(['preparation'])


class TestBaseStage:
    """Test the BaseStage class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.state_dir = self.temp_dir / ".pipeline_state"
        self.state_dir.mkdir(parents=True, exist_ok=True)
    
    def test_base_stage_creation(self):
        """Test creating a base stage"""
        stage = BaseStage(
            name="test_stage",
            description="Test stage",
            required_env="core",
            dependencies=[]
        )
        
        assert stage.name == "test_stage"
        assert stage.description == "Test stage"
        assert stage.required_env == "core"
        assert stage.dependencies == []
        assert stage.is_ci == True  # CI environment is set
    
    def test_stage_setup(self):
        """Test stage setup"""
        stage = BaseStage("test", "Test", "core", [])
        config = {"test_param": "value"}
        
        stage.setup(config, self.state_dir)
        
        assert stage.config == config
        assert stage.state_file is not None
        assert stage.state_file.parent == self.state_dir
    
    def test_stage_run_ci_safe(self):
        """Test stage run in CI-safe mode"""
        stage = BaseStage("test", "Test", "core", [])
        stage.setup({}, self.state_dir)
        
        result = stage.run()
        
        assert isinstance(result, StageResult)
        assert result.success == True
        assert 'ci_mock' in result.data.get('status', '')


class TestStageIntegration:
    """Integration tests for stage pipeline"""
    
    def setup_method(self):
        """Setup for each test"""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.manager = StageManager(project_root=self.temp_dir)
    
    def test_stage_dependency_order(self):
        """Test that stages are ordered correctly by dependencies"""
        execution_order = self.manager.execution_order
        
        # Preparation should be first (no dependencies)
        assert execution_order[0] == 'preparation'
        
        # Training should come after preparation
        prep_idx = execution_order.index('preparation')
        training_idx = execution_order.index('training')
        assert training_idx > prep_idx
        
        # Detection should come after training
        detection_idx = execution_order.index('detection')
        assert detection_idx > training_idx
        
        # OCR should come after detection
        ocr_idx = execution_order.index('ocr')
        assert ocr_idx > detection_idx
        
        # Enhancement should be last
        enhancement_idx = execution_order.index('enhancement')
        assert enhancement_idx > ocr_idx
    
    def test_stage_validation(self):
        """Test stage input validation"""
        # Get a stage that has dependencies
        training_stage = self.manager.stages['training']
        training_stage.setup({}, self.manager.state_dir)
        
        # Should return False since preparation hasn't run
        assert training_stage.validate_inputs() == False
    
    @patch('src.config.get_config')
    def test_pipeline_with_mock_config(self, mock_config):
        """Test pipeline with mocked configuration"""
        # Mock configuration
        mock_config_obj = Mock()
        mock_config_obj.config = {
            'data_root': str(self.temp_dir / 'data'),
            'paths': {}
        }
        mock_config.return_value = mock_config_obj
        
        # Run preparation stage
        result = self.manager.run_single_stage('preparation')
        assert result.success == True


class TestCSVFormatter:
    """Test CSV formatting functionality"""
    
    def test_csv_formatter_import(self):
        """Test that CSV formatter can be imported"""
        from src.output.csv_formatter import CSVFormatter, TextRegion
        
        formatter = CSVFormatter()
        assert formatter.area_grouping == True
        assert formatter.alphanumeric_sort == True
    
    def test_text_region_creation(self):
        """Test TextRegion data class"""
        from src.output.csv_formatter import TextRegion
        
        region = TextRegion(
            document="test.pdf",
            page=1,
            area_id="area_001",
            area_type="text",
            sequence="001",
            text_content="Test text",
            confidence=0.95,
            x=100.0,
            y=200.0,
            width=50.0,
            height=20.0
        )
        
        assert region.document == "test.pdf"
        assert region.text_content == "Test text"
        assert region.confidence == 0.95


class TestAreaGrouper:
    """Test area grouping functionality"""
    
    def test_area_grouper_import(self):
        """Test that area grouper can be imported"""
        from src.output.area_grouper import AreaGrouper, BoundingBox
        
        grouper = AreaGrouper()
        assert grouper.proximity_threshold == 100.0
        assert grouper.symbol_association_threshold == 200.0
    
    def test_bounding_box(self):
        """Test BoundingBox functionality"""
        from src.output.area_grouper import BoundingBox
        
        bbox1 = BoundingBox(0, 0, 100, 100)
        bbox2 = BoundingBox(50, 50, 150, 150)
        
        assert bbox1.center == (50.0, 50.0)
        assert bbox1.area == 10000.0
        assert bbox1.distance_to(bbox2) > 0
        assert bbox1.overlaps_with(bbox2) == True


if __name__ == "__main__":
    pytest.main([__file__])
