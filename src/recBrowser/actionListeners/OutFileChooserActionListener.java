package recBrowser.actionListeners;

import recBrowser.BrowserSettings;

import java.io.File;

public class OutFileChooserActionListener extends FileChooserActionListener {
	@Override
	protected void saveFilePath(File file) {
		BrowserSettings.outFilePath = file.getPath();
	}
}
