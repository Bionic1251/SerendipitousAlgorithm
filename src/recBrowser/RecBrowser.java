package recBrowser;

import recBrowser.actionListeners.*;

import javax.swing.*;
import java.awt.*;

public class RecBrowser {
	private static void createAndShowGUI() {
		JFrame frame = new JFrame("RecBrowser");
		frame.setPreferredSize(new Dimension(700, 700));
		frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
		frame.getContentPane().setLayout(new BoxLayout(frame.getContentPane(), BoxLayout.Y_AXIS));

		JPanel fileChooserPanel = new JPanel();
		fileChooserPanel.setLayout(new BoxLayout(fileChooserPanel, BoxLayout.X_AXIS));
		JButton testFileButton = new JButton("Select test file");
		testFileButton.addActionListener(new TestFileChooserActionListener());
		fileChooserPanel.add(testFileButton);
		JButton trainFileButton = new JButton("Select trainForEachUser file");
		trainFileButton.addActionListener(new TrainFileChooserActionListener());
		fileChooserPanel.add(trainFileButton);
		JButton outFileButton = new JButton("Select out file");
		outFileButton.addActionListener(new OutFileChooserActionListener());
		fileChooserPanel.add(outFileButton);
		JButton checkFiles = new JButton("Check files");
		fileChooserPanel.add(checkFiles);

		JPanel userSelectorPanel = new JPanel();
		userSelectorPanel.setLayout(new BoxLayout(userSelectorPanel, BoxLayout.X_AXIS));
		JList algList = getJList();
		userSelectorPanel.add(getJScroller(algList));
		JList userList = getJList();
		userSelectorPanel.add(getJScroller(userList));
		checkFiles.addActionListener(new CheckActionListener(algList, userList));
		JButton selectUserButton = new JButton("Select user");
		userSelectorPanel.add(selectUserButton);

		JPanel userPanel = new JPanel();
		userPanel.setLayout(new BoxLayout(userPanel, BoxLayout.X_AXIS));
		selectUserButton.addActionListener(new SelectUserActionListener(algList, userList, userPanel));

		frame.getContentPane().add(fileChooserPanel);
		frame.getContentPane().add(userSelectorPanel);
		frame.getContentPane().add(userPanel);

		frame.pack();
		frame.setVisible(true);
	}

	private static JList getJList() {
		JList algList = new JList();
		algList.setSelectionMode(ListSelectionModel.SINGLE_SELECTION);
		algList.setLayoutOrientation(JList.VERTICAL);
		return algList;
	}

	private static JScrollPane getJScroller(JList jList) {
		JScrollPane algListScroller = new JScrollPane(jList);
		algListScroller.setMaximumSize(new Dimension(100, 100));
		return algListScroller;
	}

	public static void main(String[] args) {
		javax.swing.SwingUtilities.invokeLater(new Runnable() {
			public void run() {
				createAndShowGUI();
			}
		});
	}
}
