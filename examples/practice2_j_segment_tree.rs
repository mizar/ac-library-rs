// Check Problem Statement via https://atcoder.jp/contests/practice2/tasks/practice2_j
use ac_library_rs::{Max, Segtree};
use std::io::prelude::*;
use std::iter::FromIterator;

#[allow(clippy::many_single_char_names)]
fn main() {
    let mut buf = String::new();
    std::io::stdin().read_to_string(&mut buf).unwrap();
    let mut input = buf.split_whitespace();

    let n: usize = input.next().unwrap().parse().unwrap();
    let q: usize = input.next().unwrap().parse().unwrap();
    let mut segtree = Segtree::<Max<i32>>::from_iter(
        std::iter::once(0).chain((1..=n).map(|_| input.next().unwrap().parse().unwrap())),
    );
    for _ in 0..q {
        match input.next().unwrap().parse().unwrap() {
            1 => {
                let x = input.next().unwrap().parse().unwrap();
                let v = input.next().unwrap().parse().unwrap();
                segtree.set(x, v);
            }
            2 => {
                let l = input.next().unwrap().parse().unwrap();
                let r: usize = input.next().unwrap().parse().unwrap();
                println!("{}", segtree.prod(l, r + 1));
            }
            3 => {
                let x = input.next().unwrap().parse().unwrap();
                let v = input.next().unwrap().parse().unwrap();
                println!("{}", segtree.max_right(x, |a| a < &v));
            }
            _ => {}
        }
    }
}
