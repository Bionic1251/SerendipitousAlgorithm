package recBrowser.actionListeners;

import recBrowser.BrowserSettings;

import java.io.File;

public class TestFileChooserActionListener extends FileChooserActionListener {
	@Override
	protected void saveFilePath(File file) {
		BrowserSettings.testFilePath = file.getPath();
	}
}
