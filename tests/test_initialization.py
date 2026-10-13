import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from memory import config_manager, memory_manager
from core import gemini
from dashboard import server


class TestInitializationAndCore(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.orig_config_dir = config_manager.CONFIG_DIR
        self.orig_config_file = config_manager.CONFIG_FILE
        self.orig_memory_path = memory_manager.MEMORY_PATH

        # Redirect paths to temporary directory
        config_manager.CONFIG_DIR = Path(self.test_dir) / "config"
        config_manager.CONFIG_FILE = config_manager.CONFIG_DIR / "api_keys.json"
        memory_manager.MEMORY_PATH = Path(self.test_dir) / "memory" / "long_term.json"

    def tearDown(self):
        config_manager.CONFIG_DIR = self.orig_config_dir
        config_manager.CONFIG_FILE = self.orig_config_file
        memory_manager.MEMORY_PATH = self.orig_memory_path
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_config_manager_initialization(self):
        config_manager.ensure_config_dir()
        self.assertTrue(config_manager.CONFIG_DIR.exists())

        # Test save and load API keys
        config_manager.save_api_keys("AIzaSyTestKey1234567890123")
        keys = config_manager.load_api_keys()
        self.assertEqual(keys.get("gemini_api_key"), "AIzaSyTestKey1234567890123")
        self.assertTrue(config_manager.is_configured())

        # Test assistant settings defaults
        self.assertEqual(config_manager.get_assistant_name(), "JARVIS")
        config_manager.save_assistant_config("STARK", "Tony")
        self.assertEqual(config_manager.get_assistant_name(), "STARK")
        self.assertEqual(config_manager.get_user_name(), "Tony")

        # Test voice settings
        self.assertEqual(config_manager.get_voice(), "Charon")
        config_manager.save_voice("Puck")
        self.assertEqual(config_manager.get_voice(), "Puck")

        # Test HUD style
        self.assertEqual(config_manager.get_hud_style(), "face")
        config_manager.save_hud_style("core")
        self.assertEqual(config_manager.get_hud_style(), "core")

    def test_memory_manager_initialization(self):
        mem = memory_manager.load_memory()
        self.assertIn("identity", mem)
        self.assertIn("preferences", mem)

        # Test update memory
        memory_manager.update_memory({
            "identity": {"name": {"value": "Fatih"}},
            "preferences": {"favorite_color": {"value": "Cyan"}}
        })
        updated = memory_manager.load_memory()
        self.assertEqual(updated["identity"]["name"]["value"], "Fatih")

        # Test prompt formatting & search
        prompt_block = memory_manager.format_memory_for_prompt(updated)
        self.assertIn("Fatih", prompt_block)

        search_res = memory_manager.search_memory("Cyan")
        self.assertIn("Cyan", search_res)

        # Test session memory summary saving and popping
        memory_manager.save_session_summary("Discussed project deployment", "English")
        popped = memory_manager.pop_last_session()
        self.assertIsNotNone(popped)
        self.assertEqual(popped.get("summary"), "Discussed project deployment")

    def test_gemini_ladder(self):
        self.assertTrue(gemini.is_quota_error("429 RESOURCE_EXHAUSTED"))
        self.assertTrue(gemini.is_unavailable_error("504 DEADLINE_EXCEEDED"))
        self.assertTrue(gemini.is_gone_error("404 Not Found"))

        model = gemini.live_model()
        self.assertIn("gemini", model)

    def test_dashboard_server_initialization(self):
        dash = server.DashboardServer()
        key = dash.new_key()
        self.assertEqual(len(key), 6)
        if server._DEPS_OK:
            self.assertIsNotNone(dash.app)
        else:
            self.assertIsNone(dash.app)


if __name__ == "__main__":
    unittest.main()
