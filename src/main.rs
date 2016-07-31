extern crate rand;
use rand::{thread_rng, sample};
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

enum Piece {
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

#[derive(Clone)]
#[derive(Copy)]
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

pub struct BoardState {
    squares: [i8; 64],
    to_play: Side
}

impl BoardState {
    fn get_ss(&self, xy : Coord) -> SquareState {
        let ival: i8 = self.squares[xy.as_index ()];
        return SquareState::from_i8(ival);
    }

    fn print_as_i8(&self) {
        for y in 0..8 {
            for x in 0..8 {
                print!("{}{}: {:2}  ", x, y, self.squares[(y*8) + x]);
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

    fn set_start_position(&mut self) {
        for y in 0..8 {
            for x in 0..8 {
                self.squares[(y * 8) + x] = SquareState::Empty.as_i8();
            }
        }
        self.set_occupied(0, 0, Side::White, Piece::Rook);
        self.set_occupied(1, 0, Side::White, Piece::Knight);
        self.set_occupied(2, 0, Side::White, Piece::Bishop);
        self.set_occupied(3, 0, Side::White, Piece::Queen);
        self.set_occupied(4, 0, Side::White, Piece::King);
        self.set_occupied(5, 0, Side::White, Piece::Bishop);
        self.set_occupied(6, 0, Side::White, Piece::Knight);
        self.set_occupied(7, 0, Side::White, Piece::Rook);

        for x in 0..8 {
            self.set_occupied(x, 1, Side::White, Piece::Pawn);
            self.set_occupied(x, 6, Side::Black, Piece::Pawn);
        }

        self.set_occupied(0, 7, Side::Black, Piece::Rook);
        self.set_occupied(1, 7, Side::Black, Piece::Knight);
        self.set_occupied(2, 7, Side::Black, Piece::Bishop);
        self.set_occupied(3, 7, Side::Black, Piece::Queen);
        self.set_occupied(4, 7, Side::Black, Piece::King);
        self.set_occupied(5, 7, Side::Black, Piece::Bishop);
        self.set_occupied(6, 7, Side::Black, Piece::Knight);
        self.set_occupied(7, 7, Side::Black, Piece::Rook);

        self.to_play = Side::White;
    }

    fn set_square(&mut self, x: usize, y: usize, ss: SquareState) {
        self.squares[(y * 8) + x] = ss.as_i8();
    }

    fn set_occupied(&mut self, x: usize, y: usize,
                    side: Side, piece: Piece) {
        let sp = SidedPiece {side: side, piece: piece};
        let ss = SquareState::Occupied(sp);
        self.set_square (x, y, ss);
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
        println!("");
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
        if self.to_play == Side::White {
            dir = 1;
        } else {
            dir = -1;
        }
        // Standard move:
        let to = Coord {x: from.x, y: from.y + dir};
        if to.is_valid() && self.is_unoccupied(to) {
            moves.push(Move {from: from, to: to});

            // Initial double-move:
            let to = Coord {x: from.x, y: from.y + dir + dir};
            if to.is_valid() && self.is_unoccupied(to) {
                moves.push(Move {from: from, to: to});
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

    fn apply_move(&self, m: &Move, out: &mut BoardState) {
        out.squares = self.squares;

        let from_index = m.from.as_index();
        let to_index = m.to.as_index();

        out.squares[from_index] = SquareState::Empty.as_i8();
        out.squares[to_index] = self.squares[from_index];
        // TODO: promotion

        out.to_play = self.to_play.invert ();
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
}

fn main() {
    let mut pos = BoardState {squares : [0; 64], to_play:Side::White};
    pos.set_start_position();
    pos.print();
    pos.print_as_i8();
    pos.print_unicode();
    loop {
        // Pick random moves:
        let mut rng = thread_rng();
        let mut moves = Vec::new();
        pos.get_moves(&mut moves);
        if moves.len() == 0 {
            println!("No valid moves");
            break;
            // TODO: handle checkmate
        }
        let ref m = sample(&mut rng, moves, 1)[0];
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
        let mut newpos = BoardState {squares : [0; 64], to_play:Side::White}; // FIXME
        pos.apply_move(m, &mut newpos);
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
        assert_eq!(Coord {x: 0, y: 0}.name(), "a1");
        assert_eq!(Coord {x: 7, y: 0}.name(), "h1");
        assert_eq!(Coord {x: 0, y: 7}.name(), "a8");
        assert_eq!(Coord {x: 7, y: 7}.name(), "h8");
    }

    #[test]
    fn start_position() {
        let mut pos = BoardState {squares : [0; 64], to_play:Side::White};
        pos.set_start_position();

        assert_eq!(pos.to_play, Side::White);

        let mut moves = Vec::new();
        pos.get_moves(&mut moves);
        assert_eq!(moves.len(), 20);
    }
}
