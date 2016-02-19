package recBrowser.actionListeners;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.File;

public abstract class FileChooserActionListener implements ActionListener {
	@Override
	public void actionPerformed(ActionEvent e) {
		JFileChooser fileChooser = new JFileChooser();
		fileChooser.setCurrentDirectory(new File(System.getProperty("user.dir")));
		int ret = fileChooser.showDialog(null, "Open");
		if (ret == JFileChooser.APPROVE_OPTION) {
			saveFilePath(fileChooser.getSelectedFile());
		}
	}

	abstract protected void saveFilePath(File file);
}
