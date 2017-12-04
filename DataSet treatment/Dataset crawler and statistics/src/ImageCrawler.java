import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.MalformedURLException;
import java.util.ArrayList;
import java.util.HashMap;

public class ImageCrawler {
	public static void main(String args[]){		
		String csv = "F:\\dress_patterns (1).csv";
		
		CSVReader csvReader = new CSVReader(csv);
		HashMap<String,ArrayList<String>> csvFile = csvReader.readCSV(",", true);
		
		ImageDownloader imageDownloader = new ImageDownloader();

		ArrayList<String> ids = csvFile.get("id");
		ArrayList<String> urls = csvFile.get("url");
		ArrayList<String> category = csvFile.get("cat");
		
		for(int i = 13035; i < ids.size(); i++){
			String curId = ids.get(i);
			String curUrl = urls.get(i);
			String curCategory = category.get(i);
			
			String imageName = curId + "_" + curCategory + ".jpg";
			String baseDir = "F:\\CLOTHS_DATA\\";
			
			try {
				imageDownloader.downloadImage(curUrl, baseDir + imageName);
				System.out.println("Downloaded: " + (i + 1) + " from " + ids.size());
			} catch (MalformedURLException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (FileNotFoundException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
	}
}
