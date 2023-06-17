use csv::ReaderBuilder;
use little_learner::auto_diff::RankedDifferentiable;
use little_learner::scalar::Scalar;
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

impl<A> Iris<A> {
    pub fn one_hot(&self) -> (RankedDifferentiable<A, 1>, RankedDifferentiable<A, 1>)
    where
        A: Clone + Zero + One,
    {
        let vec = vec![
            RankedDifferentiable::of_scalar(Scalar::make(self.petal_length.clone())),
            RankedDifferentiable::of_scalar(Scalar::make(self.petal_width.clone())),
            RankedDifferentiable::of_scalar(Scalar::make(self.sepal_length.clone())),
            RankedDifferentiable::of_scalar(Scalar::make(self.sepal_width.clone())),
        ];

        let mut one_hot = vec![A::zero(); 3];
        one_hot[self.class as usize] = A::one();
        let one_hot = one_hot
            .iter()
            .map(|x| RankedDifferentiable::of_scalar(Scalar::make(x.clone())))
            .collect();
        (
            RankedDifferentiable::of_vector(vec),
            RankedDifferentiable::of_vector(one_hot),
        )
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
