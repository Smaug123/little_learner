use csv::ReaderBuilder;
use std::io::Cursor;
use std::str::FromStr;

const IRIS_DATA: &str = include_str!("iris.csv");

#[derive(Eq, PartialEq, Debug)]
pub enum IrisType {
    Setosa,
    Versicolor,
    Virginica,
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
pub struct Iris {
    pub class: IrisType,
    pub petal_length: f32,
    pub petal_width: f32,
    pub sepal_length: f32,
    pub sepal_width: f32,
}

pub fn import() -> Vec<Iris> {
    let mut reader = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(Cursor::new(IRIS_DATA));
    let mut output = Vec::new();
    for record in reader.records() {
        let record = record.unwrap();
        let petal_length = f32::from_str(&record[0]).unwrap();
        let petal_width = f32::from_str(&record[1]).unwrap();
        let sepal_length = f32::from_str(&record[2]).unwrap();
        let sepal_width = f32::from_str(&record[3]).unwrap();
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

pub(crate) const EXPECTED_FIRST: Iris = Iris {
    class: IrisType::Setosa,
    petal_length: 5.1,
    petal_width: 3.5,
    sepal_length: 1.4,
    sepal_width: 0.2,
};

#[cfg(test)]
mod test {
    use crate::iris::import;

    #[test]
    fn first_element() {
        let irises = import();
        assert_eq!(irises[0], crate::iris::EXPECTED_FIRST);
    }
}
