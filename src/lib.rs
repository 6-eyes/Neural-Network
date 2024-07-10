use core::fmt;
use std::ops;
use rand::Rng;

#[derive(Debug)]
struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// Creates a random Matrix given the `rows` and `cols`
    pub fn random(rows: usize, cols: usize) -> Self {
        let mut rng = rand::thread_rng();
        Self {
            data: (0..rows * cols).map(|_| rng.gen_range(0f64..1f64)).collect(),
            rows,
            cols,
        }
    }

    /// Creates a new matrix given the `rows`, `cols` and the `vector`
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self {
            data,
            rows,
            cols,
        }
    }

    /// adds two matrices returning a new matrix.
    pub fn add_matrix(&self, rhs: &Self) -> Self {
        assert_eq!( self.rows, rhs.rows, "rows not equal!" );
        assert_eq!( self.cols, rhs.cols, "columns not equal!" );

        Self {
            data: self.data.iter().zip(&rhs.data).map(|(x, y)| x + y).collect::<Vec::<f64>>(),
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// subtracts two matrices returning a new matrix
    pub fn subtract_matrix(&self, rhs: &Self) -> Self {
        assert_eq!( self.rows, rhs.rows, "rows not equal!" );
        assert_eq!( self.cols, rhs.cols, "columns not equal!" );

        Self {
            data: self.data.iter().zip(&rhs.data).map(|(x, y)| x - y).collect::<Vec::<f64>>(),
            rows: self.rows,
            cols: self.cols,
        }
    }

    /// creates a new matrix with all zeros returning a new matrix
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec!{ 0f64; rows * cols },
        }
    }

    /// performs elementwise multiplication returning a new matrix
    pub fn elementwise_multiply(mut self, rhs: &Self) -> Self {
        assert_eq!( self.rows, rhs.rows, "rows not equal!" );
        assert_eq!( self.cols, rhs.cols, "columns not equal!" );

        self.data.iter_mut().zip(&rhs.data).for_each(|(x, y)| *x *= y);
        self
    }

    /// applies the closure to each element of the matrix
    pub fn map(mut self, func: fn(f64) -> f64) -> Self {
        self.data.iter_mut().for_each(|x| *x = func(*x));
        self
    }

    /// multiplies the matrix with the given matrix.
    /// **Self** * **other**
    pub fn dot_multiply(&self, rhs: &Self) -> Self {
        assert_eq!(self.cols, rhs.rows);
        Self {
            rows: self.rows,
            cols: rhs.cols,
            data: (0..self.rows).flat_map(|i| (0..rhs.cols).map(move |j| (0..self.cols).map(|k| self.data[i * self.cols + k] * rhs.data[k * rhs.cols + j]).sum::<f64>())).collect::<Vec::<f64>>(),
        }
    }

    /// returns the determinant of the matrix
    pub fn determinant(&self) -> f64 {
        todo!();
    }

    /// consumers a metrix and returning a transposed matrix
    pub fn transpose(self) -> Self {
        let r = self.rows as f64;
        let c = self.cols as f64;
        Self {
            data: (0..self.data.len()).map(|i| i as f64).map(|i| self.data[(i % c * r + (i / c).floor()) as usize]).collect(),
            rows: self.cols,
            cols: self.rows,
        }
    }
}

impl ops::Add for Matrix {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.add_matrix(&rhs)
    }
}

impl ops::Sub for Matrix {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.subtract_matrix(&rhs)
    }
}

impl ops::Mul for Matrix {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.dot_multiply(&rhs)
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.cols == other.cols && self.data == other.data
    }
}

impl fmt::Display for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for row in 0..self.rows {
            for col in 0..self.cols {
                write!(f, "{}", self.data[row * self.cols + col])?;
                if col < self.cols - 1 {
                    write!(f, "\t")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::Matrix;

    #[test]
    fn test_random_matix() {
        let rows = 20;
        let cols = 22;

        let random_matrix = Matrix::random(rows, cols);

        assert_eq!(rows, random_matrix.rows);
        assert_eq!(cols, random_matrix.cols);
        assert_eq!(rows * cols, random_matrix.data.len());

        for i in random_matrix.data {
            assert!(i >= 0.0 && i < 1.0);
        }
    }

    #[test]
    fn test_matrix_addition() {
        let rows = 4;
        let cols = 2;
        let matrix1 = Matrix::new(rows, cols, vec!{ 23f64, 11f64, 8f64, 18f64, 1f64, 5f64, 20f64, 31f64 });
        let matrix2 = Matrix::new(rows, cols, vec!{ 32f64, 11f64, 8f64, 81f64, 1f64, 5f64, 2f64, 13f64 });

        let expected = Matrix::new(rows, cols, vec!{ 55f64, 22f64, 16f64, 99f64, 2f64, 10f64, 22f64, 44f64 });
        assert_eq!(expected, matrix1 + matrix2);
    }

    #[test]
    #[should_panic]
    fn test_matrix_addition_different_dimentions() {
        let rows = 4;
        let cols = 2;
        let matrix1 = Matrix::new(rows, cols, vec!{ 23f64, 11f64, 8f64, 18f64, 1f64 });
        let matrix2 = Matrix::new(rows, cols, vec!{ 32f64, 11f64, 8f64, 81f64, 1f64, 5f64, 2f64, 13f64 });

        let _ = matrix1 + matrix2;
    }

    #[test]
    fn test_matrix_subtraction() {
        let rows = 4;
        let cols = 2;
        let matrix1 = Matrix::new(rows, cols, vec!{ 23f64, 11f64, 8f64, 18f64, 1f64, 5f64, 20f64, 31f64 });
        let matrix2 = Matrix::new(rows, cols, vec!{ 32f64, 11f64, 8f64, 81f64, 1f64, 5f64, 2f64, 13f64 });

        let expected = Matrix::new(rows, cols, vec!{ 9f64, 0f64, 0f64, 63f64, 0f64, 0f64, -18f64, -18f64 });
        assert_eq!(expected, matrix2 - matrix1);
    }

    #[test]
    #[should_panic]
    fn test_matrix_subtraction_different_dimentions() {
        let rows = 4;
        let cols = 2;
        let matrix1 = Matrix::new(rows, cols, vec!{ 23f64, 11f64, 8f64, 18f64, 1f64, 5f64, 20f64, 31f64 });
        let matrix2 = Matrix::new(rows, cols, vec!{ 32f64, 11f64, 8f64, 81f64 });

        let _ = matrix2 - matrix1;
    }

    #[test]
    fn test_zeros_matrix() {
        let rows = 4;
        let cols = 2;
        let matrix = Matrix::zeros(rows, cols);

        assert_eq!(rows, matrix.rows);
        assert_eq!(cols, matrix.cols);
        assert_eq!(vec!{ 0f64; rows * cols }, matrix.data);
    }

    #[test]
    fn test_elementwise_multiply() {
        let rows = 4;
        let cols = 2;
        let matrix1 = Matrix::new(rows, cols, vec!{ 23f64, 11f64, 8f64, 18f64, 1f64, 5f64, 20f64, 31f64 });
        let matrix2 = Matrix::new(rows, cols, vec!{ 32f64, 11f64, 8f64, 81f64, 1f64, 5f64, 2f64, 13f64 });

        let expected = Matrix::new(rows, cols, vec!{ 736f64, 121f64, 64f64, 1458f64, 1f64, 25f64, 40f64, 403f64 });
        assert_eq!(expected, matrix1.elementwise_multiply(&matrix2));
    }


    #[test]
    fn test_dot_multiplication() {
        let matrix1 = Matrix::new(2, 3, vec!{ 1f64, 2f64, 3f64, 4f64, 5f64, 6f64 });
        let matrix2 = Matrix::new(3, 2, vec!{ 7f64, 8f64, 9f64, 10f64, 11f64, 12f64 });

        let expected = Matrix::new(2, 2, vec!{ 58f64, 64f64, 139f64, 154f64 });
        assert_eq!(matrix1 * matrix2, expected);
    }

    #[test]
    fn test_mapping_elements() {
        let rows = 4;
        let cols = 2;
        let matrix = Matrix::new(rows, cols, vec!{ 23f64, 11f64, 8f64, 18f64, 1f64, 5f64, 20f64, 31f64 });
        let func = |n: f64| n / 10f64;
        let expected = Matrix::new(rows, cols, vec!{ 2.3f64, 1.1f64, 0.8f64, 1.8f64, 0.1f64, 0.5f64, 2f64, 3.1f64 });
        assert_eq!(expected, matrix.map(func));
    }

    #[test]
    fn test_matrix_transpose() {
        let rows = 4;
        let cols = 3;
        let matrix = Matrix::new(rows, cols, vec!{ 0f64, 1f64, 2f64, 3f64, 4f64, 5f64, 6f64, 7f64, 8f64, 9f64, 10f64, 11f64 });
        let expected = Matrix::new(cols, rows, vec!{ 0f64, 4f64, 8f64, 1f64, 5f64, 9f64, 2f64, 6f64, 10f64, 3f64, 7f64, 11f64 });
        assert_eq!(matrix.transpose(), expected);
    }
}
