package recBrowser.actionListeners;

import recBrowser.BrowserSettings;
import recBrowser.InitialRecReader;
import util.Settings;

import javax.swing.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class CheckActionListener implements ActionListener {
	private final InitialRecReader recReader = new InitialRecReader();
	private final JList userList;
	private final JList algList;

	public CheckActionListener(JList algList, JList userList) {
		this.algList = algList;
		this.userList = userList;
	}

	@Override
	public void actionPerformed(ActionEvent e) {
		recReader.readFile(BrowserSettings.outFilePath);
		algList.setListData(recReader.getAlgSet().toArray());
		userList.setListData(recReader.getUserSet().toArray());
	}
}
