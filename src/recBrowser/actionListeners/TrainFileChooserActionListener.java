package recBrowser.actionListeners;

import recBrowser.BrowserSettings;

import java.io.File;

public class TrainFileChooserActionListener extends FileChooserActionListener {
	@Override
	protected void saveFilePath(File file) {
		BrowserSettings.trainFilePath = file.getPath();
	}
}
