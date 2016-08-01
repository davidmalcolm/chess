extern crate rand;
use rand::{thread_rng, sample, ThreadRng};
use std::cmp;
use std::fmt;


#[derive(PartialEq, Copy, Clone, Debug)]
pub enum Side {
    White,
    Black,
}

impl Side {
    fn invert(self) -> Side {
        match self {
            Side::White => return Side::Black,
            Side::Black => return Side::White,
        }
    }
}

pub enum Piece {
    King,
    Queen,
    Rook,
    Bishop,
    Knight,
    Pawn,
}

impl Piece {
    fn as_i8(self) -> i8 {
        let result = match self {
            Piece::King => 1,
            Piece::Queen => 2,
            Piece::Rook => 3,
            Piece::Bishop => 4,
            Piece::Knight => 5,
            Piece::Pawn => 6,
        };
        return result;
    }

    // Value in centipawns
    fn get_value(self) -> i32 {
        return match self {
            Piece::King => 1000000, // arbitrarily large
            Piece::Queen => 900,
            Piece::Rook => 500,
            Piece::Bishop => 300,
            Piece::Knight => 300,
            Piece::Pawn => 100,
        };
    }
}

struct SidedPiece {
    side: Side,
    piece: Piece,
}

impl SidedPiece {
    fn as_i8(self) -> i8 {
        let piece_i8 = self.piece.as_i8();
        return match self.side {
            Side::White => piece_i8,
            Side::Black => -piece_i8
        }
    }
    fn as_unicode(&self) -> char {
        match self.side {
            Side::White => return match self.piece {
                Piece::King => '♔',
                Piece::Queen => '♕',
                Piece::Rook => '♖',
                Piece::Bishop => '♗',
                Piece::Knight => '♘',
                Piece::Pawn => '♙'
            },
            Side::Black => return match self.piece {
                Piece::King => '♚',
                Piece::Queen => '♛',
                Piece::Rook => '♜',
                Piece::Bishop => '♝',
                Piece::Knight => '♞',
                Piece::Pawn => '♟'
            }
        }
    }
}

enum SquareState {
    Empty,
    Occupied(SidedPiece),
}

impl SquareState {
    fn as_i8(self) -> i8 {
        let result = match self {
            SquareState::Empty => 0,
            SquareState::Occupied(sp) => sp.as_i8(),
        };
        return result;
    }

    fn from_i8(mut ival : i8) -> SquareState {
        if ival == 0 {
            return SquareState::Empty;
        } else {
            let side;
            if ival > 0 {
                side = Side::White;
            } else {
                side = Side::Black;
                ival = -ival;
            }
            let piece = match ival {
                1 => Piece::King,
                2 => Piece::Queen,
                3 => Piece::Rook,
                4 => Piece::Bishop,
                5 => Piece::Knight,
                6 => Piece::Pawn,
                _ => panic! ()
            };
            let sp = SidedPiece {side: side, piece: piece};
            return SquareState::Occupied(sp);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coord
{
    x: i8,
    y: i8
}

impl Coord
{
    fn as_index(self) -> usize {
        return ((self.y * 8) + self.x) as usize;
    }

    fn is_valid(self) -> bool {
        if self.x < 0 { return false; }
        if self.x > 7 { return false; }
        if self.y < 0 { return false; }
        if self.y > 7 { return false; }
        return true;
    }

    fn name(self) -> String {
        let files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'];
        if self.x < 0 { panic!(); }
        if self.x > 7 { panic!(); }
        let file_name = files[self.x as usize];
        return format!("{}{}", file_name, self.y + 1);
    }
}

impl fmt::Display for Coord {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // The `f` value implements the `Write` trait, which is what the
        // write! macro is expecting. Note that this formatting ignores the
        // various flags provided to format strings.
        write!(f, "{}", self.name())
        //write!(f, "({}, {})", self.x, self.y)
    }
}

pub const A1 : Coord = Coord {x: 0, y: 0};
pub const B1 : Coord = Coord {x: 1, y: 0};
pub const C1 : Coord = Coord {x: 2, y: 0};
pub const D1 : Coord = Coord {x: 3, y: 0};
pub const E1 : Coord = Coord {x: 4, y: 0};
pub const F1 : Coord = Coord {x: 5, y: 0};
pub const G1 : Coord = Coord {x: 6, y: 0};
pub const H1 : Coord = Coord {x: 7, y: 0};

pub const A2 : Coord = Coord {x: 0, y: 1};
pub const B2 : Coord = Coord {x: 1, y: 1};
pub const C2 : Coord = Coord {x: 2, y: 1};
pub const D2 : Coord = Coord {x: 3, y: 1};
pub const E2 : Coord = Coord {x: 4, y: 1};
pub const F2 : Coord = Coord {x: 5, y: 1};
pub const G2 : Coord = Coord {x: 6, y: 1};
pub const H2 : Coord = Coord {x: 7, y: 1};

pub const A3 : Coord = Coord {x: 0, y: 2};
pub const B3 : Coord = Coord {x: 1, y: 2};
pub const C3 : Coord = Coord {x: 2, y: 2};
pub const D3 : Coord = Coord {x: 3, y: 2};
pub const E3 : Coord = Coord {x: 4, y: 2};
pub const F3 : Coord = Coord {x: 5, y: 2};
pub const G3 : Coord = Coord {x: 6, y: 2};
pub const H3 : Coord = Coord {x: 7, y: 2};

pub const A4 : Coord = Coord {x: 0, y: 3};
pub const B4 : Coord = Coord {x: 1, y: 3};
pub const C4 : Coord = Coord {x: 2, y: 3};
pub const D4 : Coord = Coord {x: 3, y: 3};
pub const E4 : Coord = Coord {x: 4, y: 3};
pub const F4 : Coord = Coord {x: 5, y: 3};
pub const G4 : Coord = Coord {x: 6, y: 3};
pub const H4 : Coord = Coord {x: 7, y: 3};

pub const A5 : Coord = Coord {x: 0, y: 4};
pub const B5 : Coord = Coord {x: 1, y: 4};
pub const C5 : Coord = Coord {x: 2, y: 4};
pub const D5 : Coord = Coord {x: 3, y: 4};
pub const E5 : Coord = Coord {x: 4, y: 4};
pub const F5 : Coord = Coord {x: 5, y: 4};
pub const G5 : Coord = Coord {x: 6, y: 4};
pub const H5 : Coord = Coord {x: 7, y: 4};

pub const A6 : Coord = Coord {x: 0, y: 5};
pub const B6 : Coord = Coord {x: 1, y: 5};
pub const C6 : Coord = Coord {x: 2, y: 5};
pub const D6 : Coord = Coord {x: 3, y: 5};
pub const E6 : Coord = Coord {x: 4, y: 5};
pub const F6 : Coord = Coord {x: 5, y: 5};
pub const G6 : Coord = Coord {x: 6, y: 5};
pub const H6 : Coord = Coord {x: 7, y: 5};

pub const A7 : Coord = Coord {x: 0, y: 6};
pub const B7 : Coord = Coord {x: 1, y: 6};
pub const C7 : Coord = Coord {x: 2, y: 6};
pub const D7 : Coord = Coord {x: 3, y: 6};
pub const E7 : Coord = Coord {x: 4, y: 6};
pub const F7 : Coord = Coord {x: 5, y: 6};
pub const G7 : Coord = Coord {x: 6, y: 6};
pub const H7 : Coord = Coord {x: 7, y: 6};

pub const A8 : Coord = Coord {x: 0, y: 7};
pub const B8 : Coord = Coord {x: 1, y: 7};
pub const C8 : Coord = Coord {x: 2, y: 7};
pub const D8 : Coord = Coord {x: 3, y: 7};
pub const E8 : Coord = Coord {x: 4, y: 7};
pub const F8 : Coord = Coord {x: 5, y: 7};
pub const G8 : Coord = Coord {x: 6, y: 7};
pub const H8 : Coord = Coord {x: 7, y: 7};

#[derive(PartialEq, Copy, Clone, Debug)]
struct Move {
    from: Coord,
    to: Coord,
    //piece: Piece // for handling pawn promotion
}

impl fmt::Display for Move {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}->{}", self.from, self.to)
    }
}

// Clone is only implemented for arrays of len up to 32, so we
// implement it by introducing a tuple struct wrapping the array
// of 64 values:
struct SquareArray([i8; 64]);

impl Clone for SquareArray {
  fn clone(&self) -> SquareArray {
      SquareArray(self.0)
  }
}

#[derive(Clone)]
pub struct BoardState {
    squares: SquareArray,
    to_play: Side
}

impl BoardState {
    fn get_ss(&self, xy : Coord) -> SquareState {
        let ival: i8 = self.squares.0[xy.as_index ()];
        return SquareState::from_i8(ival);
    }

    fn print_as_i8(&self) {
        for y in 0..8 {
            for x in 0..8 {
                print!("{}{}: {:2}  ", x, y, self.squares.0[(y*8) + x]);
                //print!("{:2}  ", self.squares[(y*8) + x]);
            }
            println!("");
        }
    }

    fn print(&self) {
        println!("+--+--+--+--+--+--+--+--+");
        for y in 0..8 {
            for x in 0..8 {
                print!("|");
                let ss = self.get_ss(Coord {x: x, y: y});
                match ss {
                    SquareState::Empty => print!("  "),
                    SquareState::Occupied(sp) => {
                        match sp.side {
                            Side::White => print!("W"),
                            Side::Black => print!("B")
                        }
                        match sp.piece {
                            Piece::King => print!("K"),
                            Piece::Queen => print!("Q"),
                            Piece::Rook => print!("R"),
                            Piece::Bishop => print!("B"),
                            Piece::Knight => print!("N"),
                            Piece::Pawn => print!("P")
                        }
                    }
                };
            }
            println!("|");
            println!("+--+--+--+--+--+--+--+--+");
        }
    }

    fn print_unicode(&self) {
        println!(" abcdefgh");
        for effective_y in 0..8 {
            let y = 7 - effective_y;
            print!("{}", y + 1);
            for x in 0..8 {
                let ss = self.get_ss(Coord {x: x, y: y});
                match ss {
                    SquareState::Empty => print!(" "),
                    SquareState::Occupied(sp) => {
                        print!("{}", sp.as_unicode());
                    }
                }
            }
            println!("{}", y + 1);
        }
        println!(" abcdefgh");
        match self.to_play {
            Side::White => println!("White to play"),
            Side::Black => println!("Black to play"),
        }
    }

    /// Get the starting position.
    ///
    ///   abcdefgh
    ///  8♜♞♝♛♚♝♞♜8
    ///  7♟♟♟♟♟♟♟♟7
    ///  6        6
    ///  5        5
    ///  4        4
    ///  3        3
    ///  2♙♙♙♙♙♙♙♙2
    ///  1♖♘♗♕♔♗♘♖1
    ///   abcdefgh
    ///
    fn start_position() -> BoardState {
        let mut out = BoardState::empty_board();

        out.set_occupied(A1, Side::White, Piece::Rook);
        out.set_occupied(B1, Side::White, Piece::Knight);
        out.set_occupied(C1, Side::White, Piece::Bishop);
        out.set_occupied(D1, Side::White, Piece::Queen);
        out.set_occupied(E1, Side::White, Piece::King);
        out.set_occupied(F1, Side::White, Piece::Bishop);
        out.set_occupied(G1, Side::White, Piece::Knight);
        out.set_occupied(H1, Side::White, Piece::Rook);

        for x in 0..8 {
            out.set_occupied(Coord {x:x, y:1}, Side::White, Piece::Pawn);
            out.set_occupied(Coord {x:x, y:6}, Side::Black, Piece::Pawn);
        }

        out.set_occupied(A8, Side::Black, Piece::Rook);
        out.set_occupied(B8, Side::Black, Piece::Knight);
        out.set_occupied(C8, Side::Black, Piece::Bishop);
        out.set_occupied(D8, Side::Black, Piece::Queen);
        out.set_occupied(E8, Side::Black, Piece::King);
        out.set_occupied(F8, Side::Black, Piece::Bishop);
        out.set_occupied(G8, Side::Black, Piece::Knight);
        out.set_occupied(H8, Side::Black, Piece::Rook);

        return out;
    }

    fn empty_board() -> BoardState {
        let empty = SquareState::Empty.as_i8();
        return BoardState {squares : SquareArray([empty; 64]),
                           to_play:Side::White};
    }

    fn set_square(&mut self, xy: Coord, ss: SquareState) {
        let x = xy.x as usize;
        let y = xy.y as usize;
        self.squares.0[(y * 8) + x] = ss.as_i8();
    }

    fn set_occupied(&mut self, xy: Coord, side: Side, piece: Piece) {
        let sp = SidedPiece {side: side, piece: piece};
        let ss = SquareState::Occupied(sp);
        self.set_square (xy, ss);
    }

    fn get_moves(&self, moves: &mut Vec<Move>) {
        // TODO: convert to an iterator/coroutine
        // TODO: handle being in check (filter the moves?)
        for y in 0..8 {
            for x in 0..8 {
                let from = Coord {x: x, y: y};
                let ss = self.get_ss(from);
                match ss {
                    SquareState::Empty => (),
                    SquareState::Occupied(sp) =>
                        if sp.side == self.to_play {
                            self.get_moves_from(sp, from, moves)
                        }
                }
            }
        }
    }

    fn get_moves_from(&self, sp: SidedPiece, from: Coord,
                      moves: &mut Vec<Move>) {
        //println!("{}: {}", from, sp.as_unicode());
        let side = sp.side;
        match sp.piece {
            Piece::King => self.get_king_moves(from, side, moves),
            Piece::Queen => self.get_queen_moves(from, side, moves),
            Piece::Rook => self.get_rook_moves(from, side, moves),
            Piece::Bishop => self.get_bishop_moves(from, side, moves),
            Piece::Knight => self.get_knight_moves(from, side, moves),
            Piece::Pawn => self.get_pawn_moves(from, side, moves),
        };
    }

    fn get_king_moves(&self, from: Coord, side: Side, moves: &mut Vec<Move>) {
        self.consider_move(from, -1, -1, side, moves);
        self.consider_move(from,  0, -1, side, moves);
        self.consider_move(from,  1, -1, side, moves);
        self.consider_move(from, -1,  0, side, moves);
        self.consider_move(from,  1,  0, side, moves);
        self.consider_move(from, -1,  1, side, moves);
        self.consider_move(from,  0,  1, side, moves);
        self.consider_move(from,  1,  1, side, moves);
        // TODO: castling
    }

    fn get_queen_moves(&self, from: Coord, side: Side, moves: &mut Vec<Move>) {
        self.get_rook_moves (from, side, moves);
        self.get_bishop_moves (from, side, moves);
    }

    fn get_rook_moves(&self, from: Coord, side: Side, moves: &mut Vec<Move>) {
        self.get_moves_in_direction(from,  0, -1, side, moves);
        self.get_moves_in_direction(from,  0,  1, side, moves);
        self.get_moves_in_direction(from, -1,  0, side, moves);
        self.get_moves_in_direction(from,  1,  0, side, moves);
    }

    fn get_bishop_moves(&self, from: Coord, side: Side, moves: &mut Vec<Move>) {
        self.get_moves_in_direction(from, -1, -1, side, moves);
        self.get_moves_in_direction(from,  1, -1, side, moves);
        self.get_moves_in_direction(from, -1,  1, side, moves);
        self.get_moves_in_direction(from,  1,  1, side, moves);
    }

    fn get_moves_in_direction(&self, from: Coord, delta_x: i8, delta_y: i8,
                              side: Side, moves: &mut Vec<Move>) {
        let mut to = from;
        loop {
            to.x += delta_x;
            to.y += delta_y;
            if self.can_move_to(to, side) {
                moves.push(Move {from: from, to: to});
            } else {
                return;
            }
        }
    }

    fn consider_move(&self, from: Coord, delta_x: i8, delta_y: i8,
                      side: Side, moves: &mut Vec<Move>) {
        let to = Coord {x: from.x + delta_x, y: from.y + delta_y};
        if self.can_move_to(to, side) {
            moves.push(Move {from: from, to: to});
        }
    }

    fn get_knight_moves(&self, from: Coord, side: Side, moves: &mut Vec<Move>) {
        self.consider_move(from, -2, -1, side, moves);
        self.consider_move(from,  2, -1, side, moves);
        self.consider_move(from, -2,  1, side, moves);
        self.consider_move(from,  2,  1, side, moves);
        self.consider_move(from, -1, -2, side, moves);
        self.consider_move(from,  1, -2, side, moves);
        self.consider_move(from, -1,  2, side, moves);
        self.consider_move(from,  1,  2, side, moves);
    }

    fn get_pawn_moves(&self, from: Coord, side: Side, moves: &mut Vec<Move>) {
        let dir;
        let initial_rank;
        if side == Side::White {
            dir = 1;
            initial_rank = 1;
        } else {
            dir = -1;
            initial_rank = 6;
        }
        // Standard move:
        let to = Coord {x: from.x, y: from.y + dir};
        if to.is_valid() && self.is_unoccupied(to) {
            moves.push(Move {from: from, to: to});

            // Initial double-move:
            if from.y == initial_rank {
                let to = Coord {x: from.x, y: from.y + dir + dir};
                if to.is_valid() && self.is_unoccupied(to) {
                    moves.push(Move {from: from, to: to});
                }
            }
        }

        self.consider_pawn_capture(from, side, from.x - 1, to.y, moves);
        self.consider_pawn_capture(from, side, from.x + 1, to.y, moves);

        // TODO: promotion
    }

    fn consider_pawn_capture(&self, from: Coord, side: Side, to_x: i8,
                             to_y: i8, moves: &mut Vec<Move>) {
        let to = Coord {x: to_x, y: to_y};
        if !to.is_valid () {
            return;
        }
        let to_ss = self.get_ss(to);
        match to_ss {
            SquareState::Empty => return, // TODO: en passant
            SquareState::Occupied(to_sp) => {
                if to_sp.side == side {
                    return;
                }
            }
        }

        // Valid capture:
        moves.push(Move {from: from, to: to});
    }


    fn is_unoccupied(&self, xy: Coord) -> bool {
        let ss = self.get_ss(xy);
        match ss {
            SquareState::Empty => return true,
            SquareState::Occupied(_) => return false
        };
    }

    fn can_move_to(&self, xy: Coord, side: Side) -> bool {
        if !xy.is_valid () {
            return false;
        }

        let to_ss = self.get_ss(xy);
        match to_ss {
            SquareState::Empty => return true,
            SquareState::Occupied(to_sp) => (
                return to_sp.side != side),
        }
    }

    fn apply_move(&self, m: &Move) -> BoardState {
        let mut out = self.clone();

        let from_index = m.from.as_index();
        let to_index = m.to.as_index();

        out.squares.0[from_index] = SquareState::Empty.as_i8();
        out.squares.0[to_index] = self.squares.0[from_index];
        // TODO: promotion

        out.to_play = self.to_play.invert ();

        return out;
    }

    fn get_value(&self) -> i32 {
        let mut value :i32 = 0;
        for y in 0..8 {
            for x in 0..8 {
                let xy = Coord {x: x, y: y};
                let ss = self.get_ss(xy);
                match ss {
                    SquareState::Empty => (),
                    SquareState::Occupied(sp) => {
                        let piece_val = sp.piece.get_value();
                        if sp.side == Side::White {
                            value += piece_val;
                        } else {
                            value -= piece_val;
                        }
                    }
                }
            }
        }
        return value;
    }

    // Naive minimax implementation
    fn minimax(&self, depth: u32, side: Side) -> i32 {
        if depth == 0 { return self.get_value(); }

        let mut moves = Vec::new();
        self.get_moves(&mut moves);

        match side {
            Side::White => {
                let mut best_value = i32::min_value();
                for m in moves {
                    let newpos = self.apply_move(&m);
                    let v = newpos.minimax(depth - 1, Side::Black);
                    best_value = cmp::max(best_value, v);
                }
                return best_value;
            },
            Side::Black => {
                let mut best_value = i32::max_value();
                for m in moves {
                    let newpos = self.apply_move(&m);
                    let v = newpos.minimax(depth - 1, Side::White);
                    best_value = cmp::min(best_value, v);
                }
                return best_value;
            },
        }
    }
}

trait Player {
    /// Return Some(move), or None for surrender/stalemate
    fn play(&mut self, &BoardState) -> Option<Move>;
}

struct RandomPlayer {
    rng: rand::ThreadRng
}

impl RandomPlayer {
    fn new() -> RandomPlayer {
        RandomPlayer {rng: thread_rng()}
    }
}

impl Player for RandomPlayer {
    fn play(&mut self, bs: &BoardState) -> Option<Move> {
        let mut moves = Vec::new();
        bs.get_moves(&mut moves);
        if moves.len() == 0 {
            return None;
            // TODO: handle checkmate/stalemate
        }
        let m : Move = sample(&mut self.rng, moves, 1)[0].clone();
        return Some(m);
    }
}

struct MinimaxPlayer {
    depth: u32
}

impl Player for MinimaxPlayer {
    fn play(&mut self, bs: &BoardState) -> Option<Move> {
        let mut moves = Vec::new();
        bs.get_moves(&mut moves);
        let mut best_move = None;
        match bs.to_play {
            Side::White => {
                let mut best_value = i32::min_value();
                for m in moves {
                    let newpos = bs.apply_move(&m);
                    let v = newpos.minimax(self.depth, newpos.to_play);
                    if v > best_value {
                        best_value = v;
                        best_move = Some(m);
                    }
                }
            },
            Side::Black => {
                let mut best_value = i32::max_value();
                for m in moves {
                    let newpos = bs.apply_move(&m);
                    let v = newpos.minimax(self.depth, newpos.to_play);
                    println!("{}: move has value: {}", m, v);
                    if v < best_value {
                        best_value = v;
                        best_move = Some(m);
                    }
                }
                println!("best move has value: {}", best_value);
            },
        }
        return best_move;
    }
}

fn main() {
    let mut pos = BoardState::start_position();
    pos.print();
    pos.print_as_i8();
    pos.print_unicode();

    let mut white = RandomPlayer::new();
    let mut black = MinimaxPlayer {depth: 3};

    loop {
        // Let active player select a move.
        let mm = match pos.to_play {
            Side::White => white.play(&pos),
            Side::Black => black.play(&pos)
        };
        let m;
        match mm {
            Some(m1) => m = m1,
            None => {
                println!("No valid moves");
                // TODO: handle checkmate/stalemate
                break;
            }
        }

        // Carry out the move.
        let ss = pos.get_ss(m.from);
        match ss {
            SquareState::Empty => panic!(),
            SquareState::Occupied(sp) =>
                print!("{}: {}", sp.as_unicode(), m)
        }
        let to_ss = pos.get_ss(m.to);
        match to_ss {
            SquareState::Occupied(sp) =>
                print!(" captures {}", sp.as_unicode()),
            _ => {}
        }
        print!("\n\n");
        let newpos = pos.apply_move(&m);
        newpos.print_unicode();
        pos = newpos;
        println!("Value: {}", pos.get_value())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coord_is_valid() {
        // Some valid coords:
        assert!(Coord {x: 0, y: 0}.is_valid());
        assert!(Coord {x: 7, y: 0}.is_valid());
        assert!(Coord {x: 0, y: 7}.is_valid());
        assert!(Coord {x: 7, y: 7}.is_valid());

        // Some invalid coords:
        assert!(!Coord {x: -1, y:  0}.is_valid());
        assert!(!Coord {x:  8, y:  0}.is_valid());
        assert!(!Coord {x:  0, y: -1}.is_valid());
        assert!(!Coord {x:  0, y:  8}.is_valid());
    }

    #[test]
    fn coord_names() {
        assert_eq!(A1.name(), "a1");
        assert_eq!(H1.name(), "h1");
        assert_eq!(A8.name(), "a8");
        assert_eq!(H8.name(), "h8");
    }

    #[test]
    fn start_position() {
        let pos = BoardState::start_position();

        assert_eq!(pos.to_play, Side::White);

        let mut moves = Vec::new();
        pos.get_moves(&mut moves);
        assert_eq!(moves.len(), 20);
    }

    #[test]
    ///
    fn pawn_moves() {
        let mut pos = BoardState::empty_board();
        pos.set_occupied(B2, Side::White, Piece::Pawn);
        pos.set_occupied(D3, Side::White, Piece::Pawn);
        pos.set_occupied(B7, Side::Black, Piece::Pawn);
        pos.set_occupied(D6, Side::Black, Piece::Pawn);
        //pos.print_unicode();
        /*  abcdefgh
           8        8
           7 ♟      7
           6   ♟    6
           5        5
           4        4
           3   ♙    3
           2 ♙      2
           1        1
            abcdefgh  */
        // b2: white pawn in initial rank
        let mut moves = Vec::new();
        pos.get_pawn_moves(B2, Side::White, &mut moves);
        assert_eq!(moves.len(), 2);
        assert_eq!(moves[0].to, B3);
        assert_eq!(moves[1].to, B4);

        // d3: white pawn not in initial rank
        let mut moves = Vec::new();
        pos.get_pawn_moves(D3, Side::White, &mut moves);
        assert_eq!(moves.len(), 1);
        assert_eq!(moves[0].to, D4);

        // b7: black pawn in initial rank
        let mut moves = Vec::new();
        pos.get_pawn_moves(B7, Side::Black, &mut moves);
        assert_eq!(moves.len(), 2);
        assert_eq!(moves[0].to, B6);
        assert_eq!(moves[1].to, B5);

        // d6: black pawn not in initial rank
        let mut moves = Vec::new();
        pos.get_pawn_moves(D6, Side::Black, &mut moves);
        assert_eq!(moves.len(), 1);
        assert_eq!(moves[0].to, D5);
    }
}
