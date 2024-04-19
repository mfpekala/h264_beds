use std::fs;

fn main() {
    let contents =
        fs::read_to_string("vids/video.mp4").expect("Should have been able to read the file");
    println!("contents: {:?}", &contents[0..100])
}
