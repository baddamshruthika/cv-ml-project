import java.util.Scanner;
import java.util.HashSet;
import java.util.Arrays;

public class Solution {
    public static void main(String[] args) {
        Scanner scanner = new Scanner(System.in);
        
        // Read the entire line of input
        if (!scanner.hasNextLine()) {
            return;
        }
        String input = scanner.nextLine();
        
        // Define the set of excluded ASCII values (vowels)
        HashSet<Integer> excludedAscii = new HashSet<>(Arrays.asList(
            97, 101, 105, 111, 117, 65, 69, 73, 79, 85
        ));
        
        StringBuilder output = new StringBuilder();
        
        // Process each character in the input string
        for (int i = 0; i < input.length(); i++) {
            char ch = input.charAt(i);
            int ascii = (int) ch;
            
            // Check if the character is in the exclusion list or is a space
            if (excludedAscii.contains(ascii) || ch == ' ') {
                output.append(ch);
            } else {
                // Double the character with '#' in between
                output.append(ch).append('#').append(ch);
            }
        }
        
        // Print the final encoded string
        System.out.println(output.toString());
        
        scanner.close();
    }
}
