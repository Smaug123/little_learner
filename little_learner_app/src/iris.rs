use csv::ReaderBuilder;
use little_learner::traits::{One, Zero};
use std::fmt::Debug;
use std::io::Cursor;
use std::str::FromStr;

const IRIS_DATA: &str = include_str!("iris.csv");

#[derive(Eq, PartialEq, Debug, Clone, Copy)]
pub enum IrisType {
    Setosa = 0,
    Versicolor = 1,
    Virginica = 2,
}

impl FromStr for IrisType {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "Iris-virginica" => Ok(IrisType::Virginica),
            "Iris-versicolor" => Ok(IrisType::Versicolor),
            "Iris-setosa" => Ok(IrisType::Setosa),
            _ => Err(String::from(s)),
        }
    }
}

#[derive(PartialEq, Debug)]
pub struct Iris<A> {
    pub class: IrisType,
    pub petal_length: A,
    pub petal_width: A,
    pub sepal_length: A,
    pub sepal_width: A,
}

pub fn import<A, B>() -> Vec<Iris<A>>
where
    A: FromStr<Err = B>,
    B: Debug,
{
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(Cursor::new(IRIS_DATA));
    let mut output = Vec::new();
    for record in reader.records() {
        let record = record.unwrap();
        let petal_length = A::from_str(&record[0]).unwrap();
        let petal_width = A::from_str(&record[1]).unwrap();
        let sepal_length = A::from_str(&record[2]).unwrap();
        let sepal_width = A::from_str(&record[3]).unwrap();
        let class = IrisType::from_str(&record[4]).unwrap();
        output.push(Iris {
            class,
            petal_length,
            petal_width,
            sepal_length,
            sepal_width,
        });
    }

    output
}

/// Returns the training xs, training ys, test xs, and test ys.
pub fn partition<A>(irises: &[Iris<A>]) -> (Vec<[A; 4]>, Vec<[A; 3]>, Vec<[A; 4]>, Vec<[A; 3]>)
where
    A: Clone + One + Zero + Copy,
{
    let training_cutoff = (irises.len() * 9) / 10;
    let mut training_xs = Vec::with_capacity(training_cutoff);
    let mut training_ys = Vec::with_capacity(training_cutoff);
    for iris in &irises[0..training_cutoff] {
        let (x, y) = iris.one_hot();
        training_xs.push(x);
        training_ys.push(y);
    }

    let mut test_xs = Vec::with_capacity(irises.len() - training_cutoff);
    let mut test_ys = Vec::with_capacity(irises.len() - training_cutoff);
    for iris in &irises[training_cutoff..] {
        let (x, y) = iris.one_hot();
        test_xs.push(x);
        test_ys.push(y);
    }

    (training_xs, training_ys, test_xs, test_ys)
}

impl<A> Iris<A> {
    pub fn one_hot(&self) -> ([A; 4], [A; 3])
    where
        A: Copy + Clone + Zero + One,
    {
        let vec = [
            self.petal_length.clone(),
            self.petal_width.clone(),
            self.sepal_length.clone(),
            self.sepal_width.clone(),
        ];

        let mut one_hot = [A::zero(); 3];
        one_hot[self.class as usize] = A::one();
        (vec, one_hot)
    }

    pub fn map<B, F>(&self, mut f: F) -> Iris<B>
    where
        F: FnMut(A) -> B,
        A: Copy,
    {
        Iris {
            class: self.class,
            petal_length: f(self.petal_length),
            petal_width: f(self.petal_width),
            sepal_length: f(self.sepal_length),
            sepal_width: f(self.sepal_width),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::iris::{import, Iris, IrisType};

    const EXPECTED_FIRST: Iris<f32> = Iris {
        class: IrisType::Setosa,
        petal_length: 5.1,
        petal_width: 3.5,
        sepal_length: 1.4,
        sepal_width: 0.2,
    };

    #[test]
    fn first_element() {
        let irises = import();
        assert_eq!(irises[0], EXPECTED_FIRST);
    }
}
