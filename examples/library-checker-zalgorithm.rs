// Check Problem Statement via https://judge.yosupo.jp/problem/zalgorithm
use ac_library_rs::string::z_algorithm;
use std::io::prelude::*;

fn main() {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf).unwrap();
    let mut input = buf.split_whitespace();

    let s = input.next().unwrap();

    print_oneline(z_algorithm(s));
}

fn print_oneline<I: IntoIterator<Item = T>, T: std::fmt::Display>(values: I) {
    use std::fmt::Write;
    println!(
        "{}",
        values
            .into_iter()
            .fold(String::new(), |mut acc, cur| {
                write!(&mut acc, "{}", cur).unwrap();
                acc.push(' ');
                acc
            })
            .trim_end()
    );
}
